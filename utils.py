import copy
from itertools import zip_longest
from pathlib import Path

import librosa
import numpy as np
import scipy
import sys
import warnings
import sounddevice
import os

from matplotlib import pyplot as plt, ticker
from scipy.ndimage import uniform_filter1d

generate_random_seed = False
if generate_random_seed:
    rnd_seed = int(np.random.rand() * (2 ** 32 - 1))
else:
    # rnd_seed = 866369434
    # rnd_seed = 1220931063
    # rnd_seed = 122093106313
    rnd_seed = 122093106313 + 1
    print(f"Random seed: {rnd_seed}")
rng = np.random.default_rng(rnd_seed)
eps = np.finfo(float).eps
cmap = 'plasma'


def savefig(figure, file_name: Path, dpi=300, transparent=True):
    # If filename already exists, append a number to the end of the filename (but before extension), starting from -1
    # and increasing until a filename is found that does not exist.
    idx = 0
    file_name_stem = file_name.stem
    file_name = file_name.with_name(file_name_stem + f"_{idx}" + file_name.suffix)
    while file_name.exists():
        idx += 1
        file_name = file_name.with_name(file_name_stem + f"_{idx}" + file_name.suffix)

    # facecolor parameter determines the background color of the saved figure
    # edgecolor parameter determines the color of the border of the saved figure
    # transparent=True makes the background transparent
    sett = {'dpi': dpi, 'transparent': transparent, 'bbox_inches': 'tight'}
    if not transparent:
        sett['facecolor'] = figure.get_facecolor()
        sett['edgecolor'] = figure.get_edgecolor()

    figure.savefig(file_name, **sett)

    print(f"Figure saved as {file_name}")

    return file_name


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


# @jit(nopython=True, cache=True)
def compute_correction_term(stft_shape, overlap, complex_stft=False):
    """
    Compute the correction term to account for delay in STFT. See for example Equation 3 in
    "Fast computation of the spectral correlation" by Antoni, 2017.

    :param complex_stft: bool, if True, then the input stft to correct is complex, otherwise it is real
    :param stft_shape: shape of the stft matrix, e.g. (num_mics, num, num_frames)
    :param overlap: a number between 0 and win_len-1, where win_len = (num_freqs-1)*2
    :return:
    """

    (num_freqs, num_frames) = stft_shape

    # If only positive ("real") frequencies are available, the window length is (num_freqs-1)*2
    win_len = num_freqs if complex_stft else (num_freqs - 1) * 2
    shift_samples = win_len - overlap

    # Range is [0, 0.5) (real part). With negative frequencies, the range would be [-0.5, 0.5).
    normalized_frequencies = np.arange(-num_freqs // 2, num_freqs // 2) if complex_stft else np.arange(0, num_freqs)
    normalized_frequencies = normalized_frequencies[:, np.newaxis] / win_len

    # Compute correction term using array broadcasting. Accounts for delay in STFT.
    time_frames = np.arange(0, shift_samples * num_frames, shift_samples)[np.newaxis, :]
    correction_term = np.exp(-2j * np.pi * normalized_frequencies * time_frames)

    return correction_term


# def plot(x, ax=None, title=''):
#     """For one or multiple 1-D plots, i.e. for time-domain plots."""
#
#     if 1:
#         is_subplot = ax is not None
#         if is_subplot:
#             fig = plt.gcf()
#         else:
#             fig, ax = plt.subplots(1, 1)
#
#     if x.ndim == 2 and x.shape[1] > x.shape[0]:
#         ax.plot(x.T)
#     else:
#         ax.plot(x)
#     ax.grid(True)
#
#     if title != '':
#         ax.set_title(title)
#
#     plt.show()
#     return fig
def pad_last_dim(x, N_, prepad=False):
    assert x.ndim <= 2, "Only 1d and 2d arrays are supported."
    # Should work both for 1d and 2d arrays
    if N_ > x.shape[-1]:
        if not prepad:
            return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, N_ - x.shape[-1])])
        else:
            return np.pad(x, [(0, N_ - x.shape[-1])] + [(0, 0)] * (x.ndim - 1))
    else:
        return x


def plot(x, ax=None, titles='', fs=16000, time_axis=True, plot_config=None, subplot_height=0.8):
    """For one or multiple 1-D plots, i.e. for time-domain plots."""

    font_size = 'small'
    title_font_size = 'medium'

    """
    if isinstance(x, list) and len(x) == 1:
        x = x[0]
        if isinstance(titles, list):
            titles = titles[0]
    """

    if isinstance(x, list):
        num_plots = len(x)

        # sharex flag true if all arrays in x have the same length
        sharex = all(len(x[0]) == len(item) for item in x)

        max_len = max(item.shape[-1] for item in x)
        x = [pad_last_dim(item, max_len) for item in x]

        fig_opt = dict(figsize=(6, 0.5 + num_plots * subplot_height), layout='compressed', squeeze=False)
        fig, axes = plt.subplots(num_plots, 1, sharey='all', sharex=sharex, **fig_opt)

        for ax, audio_sample, title in zip_longest(axes.flat, x, titles):
            plot(audio_sample, ax, fs=fs, time_axis=time_axis, plot_config=plot_config)
            ax.set_ylabel("Amplitude", fontsize=font_size)

            if time_axis:
                ax.set_xlabel("Time [s]", fontsize=font_size)
                x_locations, _ = ax.get_xticks(), ax.get_xticklabels()
                labels_str = [f"{x / fs:.2f}" for x in x_locations]
                ax.set_xticks(x_locations, labels_str)
                ax.set_xlim(0, audio_sample.shape[-1])

                num_x_ticks = 10
                x_locator = ticker.MaxNLocator(num_x_ticks)  # , integer=True
                x_minor_locator = ticker.AutoMinorLocator(4)  # 4
                y_minor_locator = ticker.AutoMinorLocator(2)  # 2
                ax.xaxis.set_major_locator(x_locator)
                ax.tick_params(axis='both', labelsize=font_size)
                ax.grid(which='both')

                # Change minor ticks to show every 5. (20/4 = 5)
                if x_minor_locator is not None:
                    ax.xaxis.set_minor_locator(x_minor_locator)
                ax.yaxis.set_minor_locator(y_minor_locator)
                ax.grid(which='major', color='#CCCCCC')
                ax.xaxis.grid(which='minor', color='#CCCCCC', linestyle=':', linewidth=0.3)
                ax.yaxis.grid(which='minor', color='#CCCCCC', linestyle='-', linewidth=0.3)

            ax.set_title(title, fontsize=title_font_size)

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axes.flat:
            if sharex:
                ax.label_outer()
            ax.grid(True)

        fig.show()
        return fig

    else:
        is_subplot = ax is not None
        if is_subplot:
            fig = plt.gcf()
        else:
            fig, ax = plt.subplots(1, 1)
            ax.set_title(titles, fontsize=title_font_size)

    if plot_config is None:
        plot_config = dict()

    if x.ndim == 2 and x.shape[1] > x.shape[0]:
        ax.plot(x.T, **plot_config)
    else:
        ax.plot(x, **plot_config)

    ax.grid(True)

    if not is_subplot:
        plt.show()
        plt.pause(0.05)

    return fig


def plot_matrix(X_, title='', xy_label=('', ''), xy_ticks=None, log=True, show_figures=True,
                amp_range=(None, None), figsize=None, normalized=False):

    if np.allclose(X_, 0):
        warnings.warn(f"X with {title = } and {X_.shape = } is zero. Skipping plot.")
        return None

    X = copy.deepcopy(X_)
    X = np.atleast_2d(np.squeeze(X))
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)
    options = dict(cmap=cmap, antialiased=True)

    if amp_range[0] is not None:
        options['vmin'] = amp_range[0]
    if amp_range[1] is not None:
        options['vmax'] = amp_range[1]
    if 'vmin' in options and 'vmax' in options and options['vmin'] > options['vmax']:
        raise ValueError(f"Invalid amp_range: {amp_range}, vmin > vmax.")
    font_size = 11
    title_font_size = 14

    if normalized:
        X = X / (eps + np.max(X))

    if log:
        if X.size == 0:
            warnings.warn("X is empty. Cannot compute log.")
            return fig
        small_const = np.nanmin(np.abs(X)[np.abs(X) > 0]) / 100
        X = 10 * np.log(small_const + np.abs(X) ** 2)

    if xy_ticks is not None:
        # X, Y = np.meshgrid(*xy_ticks)
        pcm_mag = ax.pcolormesh(*xy_ticks, X, **options)
    else:
        pcm_mag = ax.pcolormesh(X, **options)

    if title == '':
        title = f"Spectral correlation"
    ax.set_title(title, fontsize=title_font_size)

    cl = fig.colorbar(pcm_mag, ax=ax)
    label = 'Magnitude (dB)' if log else 'Magnitude'
    cl.set_label(label, size=font_size)
    cl.ax.tick_params(labelsize=font_size)

    # ax.invert_yaxis()
    ax.xaxis.set_ticks_position('both')

    # Tick label size
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    if xy_label != ('', ''):  # if not empty
        ax.set_xlabel(xy_label[0], fontsize=font_size)
        ax.set_ylabel(xy_label[1], fontsize=font_size)

    if show_figures:
        fig.show()
        plt.pause(0.05)

    return fig


def fig_to_subplot(existing_fig, title, ax, xy_ticks=(None, None), xlabel='', ylabel=''):

    if existing_fig is None or not existing_fig:
        return None

    # Retrieve the image data from the existing figure
    img = existing_fig.axes[0].collections[0].get_array().data

    # Retrieve vmin and vmax from the existing figure
    vmin, vmax = existing_fig.axes[0].collections[0].get_clim()

    # Retrieve the x ticks, y ticks, color map, and labels from the existing figure
    cmap = existing_fig.axes[0].collections[0].get_cmap()

    # Display the image data in the new subplot
    if isinstance(xy_ticks, tuple) and xy_ticks[0].any() and xy_ticks[1].any():
        im = ax.pcolormesh(*xy_ticks, img, antialiased=True, vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        im = ax.pcolormesh(img, antialiased=True, vmin=vmin, vmax=vmax, cmap=cmap)

    if xlabel == '':
        xlabel = 'Cyclic freq.~$\\alpha_p$ [kHz]'

    if ylabel == '':
        ylabel = 'Freq.~$\\omega_k$ [kHz]'

    # Set the title of the subplot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return im


def generate_harmonic_signal(f0_, fs_, L_, num_harmonics_=3, frequency_error_=0.0, rnd_amplitude=False, rnd_phase=True):
    # Generate a harmonic signal with random phase and amplitude.

    freq_error = rng.uniform(-frequency_error_, frequency_error_, num_harmonics_)
    discrete_frequencies = f0_ * (np.arange(num_harmonics_) + 1 + freq_error)
    phases = rng.uniform(0, 2 * np.pi, num_harmonics_) if rnd_phase else np.zeros(num_harmonics_)
    amplitudes = rng.uniform(0.5, 1.0, num_harmonics_) if rnd_amplitude else np.ones(num_harmonics_)
    discrete_times = np.arange(L_) / fs_

    y = np.sum(amplitudes[:, None] *
               np.cos(2 * np.pi * discrete_frequencies[:, None] * discrete_times[np.newaxis, :] + phases[:, None]),
               axis=0)

    y = y / (eps + np.max(np.abs(y)))
    y -= np.mean(y)

    return y


def stft(y, fs_=16000, Nw_=512, noverlap_samples_=0, complex_stft=False, window='hann', padding=False):
    # Had to disable boundary and padded to get the same results as in paper "A faster algorithm for the calculation
    # of the fast spectral correlation" by Borghesani, 2018.

    if padding:
        padded = True
        boundary = 'zeros'
    else:
        padded = False
        boundary = None

    _, _, y_stft = scipy.signal.stft(y, fs=fs_, window=window, nperseg=Nw_, noverlap=noverlap_samples_, detrend=False,
                                     return_onesided=not complex_stft,
                                     boundary=boundary, padded=padded, axis=-1)
    return y_stft


def istft(y_stft, fs_=16000, Nw_=512, noverlap_samples_=0, complex_stft=False):
    _, y = scipy.signal.istft(y_stft, fs=fs_, window='hann', nperseg=Nw_, noverlap=noverlap_samples_,
                              input_onesided=not complex_stft)
    return y


def set_printoptions_numpy():
    """ Set numpy print options to make it easier to read. Also set pprint as default for dict() """
    desired_width = 220
    np.set_printoptions(precision=3, linewidth=desired_width, suppress=True)

    # make warnings more readable
    def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
        return '%s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)

    warnings.formatwarning = warning_on_one_line


def play(sound, fs=16000, max_length_seconds=5, normalize_flag=True, volume=0.75):
    sound_normalized = volume * normalize_volume(sound) if normalize_flag else sound
    max_length_samples = int(max_length_seconds * fs)
    blocking = True

    sound_normalized_pad = pad_last_dim(sound_normalized, int(1. * fs), prepad=True)

    if 2 <= sound_normalized.shape[0] < 10:  # multichannel input was given! Play first and last channel
        sounddevice.play(sound_normalized_pad[(0, -1), :max_length_samples].T, fs, blocking=blocking)
    else:
        sounddevice.play(np.squeeze(sound_normalized_pad)[:max_length_samples], fs, blocking=blocking)


def normalize_volume(x_samples, max_value=0.9):
    if np.max(np.abs(x_samples)) < 1e-6:
        warnings.warn(f"Skipping normalization as it would amplify numerical noise.")
        return x_samples
    else:
        return max_value * x_samples / np.max(np.abs(x_samples))


def check_create_folder(folder_name, parent_folder=None):
    if parent_folder is None:
        parent_folder = os.getcwd()
    else:
        if not parent_folder.exists():
            parent_folder.mkdir(parents=True)
    child_folder = os.path.join(parent_folder, folder_name)
    if not os.path.exists(child_folder):
        os.mkdir(child_folder)
    return child_folder


def set_plot_options(use_tex=False):
    plt.style.use('seaborn-v0_8-paper')

    if not use_tex:
        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
    else:
        """
        This might be interesting at some point
        https://github.com/Python4AstronomersAndParticlePhysicists/PythonWorkshop-ICE/tree/master/examples/use_system_latex
        """
        plt.rcParams['text.usetex'] = True
        plt.rcParams["axes.formatter.use_mathtext"] = True
        font = {'family': 'serif',
                'size': 10,
                'serif': 'cmr10'
                }
        plt.rc('font', **font)
        plt.rcParams['text.latex.preamble'] = '\\usepackage{amsmath}'  # for \text command


# markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
# linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
#           'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def plot_surface(z, x, y, title=None, xy_label=('', ''), xlim=None, ylim=None, show_figures=True):
    fig = plt.figure(figsize=(4.5, 4))
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, np.abs(z), cmap=cmap, antialiased=True, rstride=1, cstride=1)

    if xy_label != ('', ''):
        ax.set_xlabel(xy_label[0])
        ax.set_ylabel(xy_label[1])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    num_ticks_desired = 5
    spacing_options = np.array([100, 200, 250, 300, 500, 600, 700, 750, 800, 900, 1000, 1500, 2000])

    # Use spacing option that leads closest to num_ticks_desired ticks
    best_option_idx_x = np.argmin(np.abs(spacing_options * num_ticks_desired - xlim[1] + xlim[0]))
    spacing_x = spacing_options[best_option_idx_x]
    x_ticks = np.arange(xlim[0], xlim[1], spacing_x).astype(int)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)

    best_option_idx_y = np.argmin(np.abs(spacing_options * num_ticks_desired - ylim[1] + ylim[0]))
    spacing_y = spacing_options[best_option_idx_y]
    y_ticks = np.arange(ylim[0], ylim[1], spacing_y).astype(int)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)

    # Make tick labels smaller and closer to axis
    ax.tick_params(axis='x', which='major', pad=-2, labelsize=8)
    ax.tick_params(axis='y', which='major', pad=-2, labelsize=8)

    ax.set_zlabel('Magnitude (normalized)')
    ax.xaxis.labelpad = -3  # Position z-label closer to z-axis
    ax.yaxis.labelpad = -3  # Position z-label closer to z-axis
    ax.zaxis.labelpad = -10  # Position z-label closer to z-axis

    if title is not None:
        # Set title and position it close to the plot
        ax.set_title(title, pad=-20, fontsize=12, y=1)

    # Make panes transparent
    ax.xaxis.pane.fill = False  # Left pane
    ax.yaxis.pane.fill = False  # Right pane
    ax.zaxis.pane.fill = False  # Right pane

    # Remove grid lines
    ax.grid(False)

    ax.set_zticks([])
    ax.set_zticklabels([])

    # Transparent spines (axes lines). If we remove this, axes labels are not visible.
    # ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Transparent panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.view_init(elev=22, roll=0, azim=330)
    fig.subplots_adjust(wspace=0, hspace=0)

    if show_figures:
        fig.show()
        plt.pause(0.05)

    return fig


def is_debug_mode():
    if sys.gettrace() is None:
        return False
    return True


def reload(module_name):
    import importlib
    importlib.reload(module_name)


# Clips real and imag part of complex number independently
def clip_cpx(x, a_min, a_max):
    if np.iscomplexobj(x):
        x = np.clip(x.real, a_min, a_max) + 1j * np.clip(x.imag, a_min, a_max)
    else:
        x = np.clip(x, a_min, a_max)
    return x


def load_audio_file(audio_file_path, fs_, N_num_samples=-1, offset_seconds=0, smoothing_window=True):
    if not os.path.exists(audio_file_path):
        raise ValueError(f"File {audio_file_path} does not exist.")

    # Load audio file
    s, fs = librosa.load(audio_file_path, sr=fs_)

    # Offset to approximately start at the vowel
    s = s[int(offset_seconds * fs):]

    if N_num_samples == -1:
        N_num_samples = len(s)
    elif len(s) < N_num_samples:
        warnings.warn(f"Signal is too short: {len(s)} samples, but {N_num_samples} samples are needed.")
        s = np.pad(s, (0, N_num_samples - len(s)))
    s = s[:N_num_samples]

    if smoothing_window:
        win = scipy.signal.windows.tukey(N_num_samples, alpha=0.2)
        s = s * win

    if np.max(np.abs(s)) < 1e-3:
        raise ValueError("Signal is too small. Select a different segment.")

    return s, fs


def generate_uniform_filtered_process(N_samples_, low=0, high=1, ma_order=10):
    x = rng.uniform(low, high, N_samples_)
    x = uniform_filter1d(x, size=ma_order)
    return x


def generate_gaussian_filtered_process(N_samples_, mean=0., variance=1., ma_order=10):
    x = rng.normal(mean, variance, N_samples_)
    x = uniform_filter1d(x, size=ma_order)
    return x


def generate_harmonic_process(freqs_hz, N_samples, fs, amplitudes_over_time=None, phases=None,
                              smooth_edges_window=False, amp_harmonic=0.5, var_harmonic=10):
    """
    Generate a sinusoid with the given frequencies and number of samples.
    The phase is a uniform random variable.
    """

    def amp_generator_wss(mean_, variance_):
        return generate_gaussian_filtered_process(N_samples, mean=mean_, variance=variance_, ma_order=int(fs * 0.1))

    num_harmonics_ = len(freqs_hz)
    discrete_times = np.arange(N_samples) / fs

    if phases is None:
        phases = rng.uniform(-np.pi, np.pi, num_harmonics_)

    if amplitudes_over_time is None:
        amplitudes_over_time = [amp_generator_wss(amp_harmonic, var_harmonic) for _ in range(num_harmonics_)]
        amplitudes_over_time = np.array(amplitudes_over_time)

    # Broadcast to shape (num_harmonics, N_samples)
    z = np.sum(amplitudes_over_time * np.cos(2 * np.pi * freqs_hz[:, None] * discrete_times[None, :] + phases[:, None]),
               axis=0)

    if smooth_edges_window:
        win = scipy.signal.windows.tukey(N_samples, alpha=0.1)
        z = z * win

    return z
