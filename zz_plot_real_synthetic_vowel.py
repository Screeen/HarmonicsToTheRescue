import warnings
from pathlib import PosixPath, Path
import numpy as np
import scipy
import matplotlib.pyplot as plt

import manager
import utils as u
import zz_plot_real_synthetic_vowel_helper as helper
import pickle
import spectral_correlation_estimator as sce

u.set_plot_options(use_tex=True)
u.set_printoptions_numpy()

scf_dict_first_realization_phase = {}
scf_dict_avg_phase = {}
sig_prop_phase = {}
f2ds_phase = [None]
f2ds_phase_avg = [None]
f2ds_real = [None]
scf_dict_first_realization_amp = {}
scf_dict_avg_amp = {}
sig_prop_amp = {}
y_real = np.array([])
im = None
x_ticks_labels = []
y_ticks_labels = []

show_single_scf_plots = False
show_combined_scf_plots = True
show_combined_time_domain_plots = False

try_loading_data = False

compute_real_data = True
compute_avg_random_phase_data = True
compute_avg_random_amplitude_data = True

f_cutoff = 600
fs = 48000
# f0 = 100
Nw = 4096
# N = int(np.round(0.11 * fs))  # approximate N
# N = int(np.round(0.17 * fs))
N = int(np.round(0.25 * fs))
# N = int(np.round(1. * fs))
L = int(np.ceil(N / Nw))  # number of frames computed as total_length_samples / window_length_samples
N = L * Nw  # adjust N to be a multiple of Nw
R_shift_samples = Nw // 3
snr = np.inf
num_realizations = 100
amplitude_single_harmonic = 0.5  # mean of the Gaussian process which modulates the amplitude of the harmonic
print(f"{N = }, {L = }, {R_shift_samples = }, signal duration: {N / fs:.2f}s")

if compute_real_data:
    module_parent = Path(__file__).resolve().parent
    dataset_path_parent = module_parent.parent / 'datasets' / 'north_texas_vowels' / 'data'
    file_name = 'kadpal03.wav'
    if dataset_path_parent.exists():
        pp = dataset_path_parent / file_name
    else:
        warnings.warn(f"Path {dataset_path_parent} does not exist. Use sample file.")
        pp = module_parent / 'audio' / file_name

    y_real, _ = u.load_audio_file(pp, fs_=fs, offset_seconds=0.045, N_num_samples=N, smoothing_window=False)
    f0_stats, _ = manager.Manager.find_f0_from_recording(y_real, fs, 512, 2048)
    f0 = np.round(f0_stats[1])
else:
    f0 = 100
    num_harmonics = 4

num_harmonics = (f_cutoff - f0) // f0
print(f"{f0 = }, {num_harmonics = }")
assert num_harmonics > 0

dft_props = {'nfft': Nw, 'noverlap': Nw - R_shift_samples, 'window': 'hann', 'nw': Nw, 'fs': fs}
scf_cfg = {'dft_props': dft_props, 'num_harmonics': num_harmonics, 'alpha_max_hz': f_cutoff + f0 // 2,
           'conjugate_scf': False}
plot_cfg = {'amp_range': (-80, 0), 'figsize': (8, 3.5), 'show_figures': show_single_scf_plots}
h = helper.Helper()

if scf_cfg['conjugate_scf']:
    warnings.warn(f"Computing conjugate SCF, not standard SCF")

if compute_real_data:
    # Post-process real signal (low-pass filter, add noise, normalize)
    sos = scipy.signal.butter(10, f_cutoff, 'low', fs=fs, output='sos')
    y_real = scipy.signal.sosfilt(sos, y_real)
    y_real, _ = manager.Manager.add_noise_snr(y_real, snr_db=snr)
    y_real = helper.normalize(y_real)

# Generate synthetic signals: random amplitude and random phase processes
sin_gen = helper.SinusoidGenerator()
y_rnd_amplitude_list_temp = []
y_rnd_phase_list_temp = []

# Harmonic frequencies WITHOUT DC component
freqs_hz = f0 * np.arange(start=1, stop=num_harmonics + 1)

print(f"Simulating {num_realizations} realizations of the signal.")
for ii in range(1 + num_realizations):
    # In all but the first realization (ii = 0), generate a single frame of the signal to compute
    # the expected value of the SCF (ideal case)
    if ii == 0:
        win_len = Nw
        num_frames = L
    else:
        win_len = N
        num_frames = 1

    y_a, y_p = sin_gen.generate_two_harmonic_processes(freqs_hz=freqs_hz, Nw_win_length=win_len, L_num_frames=num_frames,
                                                       fs=fs, amplitude_single_harmonic=amplitude_single_harmonic)
    y_a, _ = manager.Manager.add_noise_snr(y_a, snr_db=snr)
    y_p, _ = manager.Manager.add_noise_snr(y_p, snr_db=snr)

    y_a = helper.normalize(y_a)
    y_p = helper.normalize(y_p)

    y_rnd_amplitude_list_temp.append(y_a)
    y_rnd_phase_list_temp.append(y_p)

sample_paths_list = [None] * 6
if compute_real_data:
    sample_paths_list[0] = y_real
    # sample_paths_list[1] = None

sample_paths_list[2] = y_rnd_phase_list_temp[0]
if compute_avg_random_phase_data:
    sample_paths_list[3] = np.stack(y_rnd_phase_list_temp[1:])

sample_paths_list[4] = y_rnd_amplitude_list_temp[0]
if compute_avg_random_amplitude_data:
    sample_paths_list[5] = np.stack(y_rnd_amplitude_list_temp[1:])

titles = ['Voiced speech' if compute_real_data else None,
          None,
          'WSS harmonic model',
          'WSS harmonic model (avg)' if compute_avg_random_phase_data else None,
          'Cyclostationary harmonic model',
          'Cyclostationary harmonic model (avg)' if compute_avg_random_amplitude_data else None]

# SCF plots
if show_combined_scf_plots:

    SCE = sce.SpectralCorrelationEstimator(dft_props=dft_props)
    sce_props, sig_props = helper.Helper.prepare_parameters_cyclic_spectrum_estimation(N_num_samples_=N, f0=f0,
                                                                                       **scf_cfg)

    if try_loading_data and Path('estimated_scfs_and_freqs_list.pkl').exists():
        with open('estimated_scfs_and_freqs_list.pkl', 'rb') as f:
            estimated_scfs_and_freqs_list = pickle.load(f)
    else:
        print(f"Estimating SCFs for {len(sample_paths_list)} signals.")
        # Calculate the spectral correlation functions
        estimated_scfs_and_freqs_list = []
        for yy, title in zip(sample_paths_list, titles):
            print(f"Computing SCF for {title}.")
            cyc_spectrum_dict_single_realization = SCE.run_spectral_correlation_estimators(x=yy, **sce_props)
            estimated_scfs_and_freqs_list.append(cyc_spectrum_dict_single_realization)

        # save estimated_scfs_and_freqs_list using pickle
        with open('estimated_scfs_and_freqs_list.pkl', 'wb') as f:
            import pickle
            pickle.dump(estimated_scfs_and_freqs_list, f)

    # Plot the SCFs
    print(f"Plotting SCFs.")
    existing_figs = []
    for zz in estimated_scfs_and_freqs_list:
        fig_zz, x_ticks_labels, y_ticks_labels = h.plot_2d_scf(zz, sig_props, **plot_cfg, return_tick_labels=True)
        existing_figs.extend(fig_zz)

    # Combine the SCFs plots into a single figure
    fig = plt.figure(figsize=(5, 6), layout='compressed')

    # Loop over the list of existing figures
    # The new figure will have 3 rows and 2 columns. The first row will contain the real signal and an empty spot,
    # the second row will contain the random phase and random amplitude signals, and the third row will contain the
    # average SCFs.
    for ii, (existing_fig, title) in enumerate(zip(existing_figs, titles)):
        ax = fig.add_subplot(3, 2, ii + 1)
        im = u.fig_to_subplot(existing_fig, title, ax, xy_ticks=(x_ticks_labels, y_ticks_labels))

    fig.get_axes()[1].axis('off')  # Hide top right subplot

    # The colorbar will be shared among all subplots and should be placed in the first row to the right,
    # in place of the empty spot. So it should be shifted to the left
    # Label is 'Normalize magnitude [dB]'
    if im:
        fig.colorbar(im, ax=fig.get_axes()[1], orientation='horizontal', pad=-0.75, label='Normalized magnitude [dB]')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in fig.get_axes():
        ax.label_outer()

    fig.show()
    u.check_create_folder(folder_name='figs', parent_folder=Path.cwd())
    u.savefig(figure=fig, file_name=PosixPath('figs') / '2d_scfs_real_vs_rnd_amp_vs_rnd_phase.pdf',
              transparent=True)

# Time-domain plots (waveforms)
if show_combined_time_domain_plots:
    y_rnd_amplitude = y_rnd_amplitude_list_temp[0]
    y_rnd_phase = y_rnd_phase_list_temp[0]

    signals = [y_real, y_rnd_phase, y_rnd_amplitude]
    titles = [
        'Voiced speech $s_{\\text{real}}(n)$',
        'WSS harmonic model $s_{\\text{ph}}(n)$',
        'Cyclostationary harmonic model $s_{\\text{amp}}(n)$',
    ]

    fig = u.plot(signals, titles=titles, fs=fs)
    axes = fig.get_axes()
    for ax in axes:
        ax.set_ylim(-1, 1)

    fig.set_size_inches(3.8, 4.5)

    # On the second ax (random phase), insert a vertical red line every 2048 samples
    for i in range(0, len(y_rnd_phase), Nw):
        axes[1].axvline(i, color='r', linestyle='--', linewidth=1)
    fig.show()
    u.check_create_folder(folder_name='figs', parent_folder=Path.cwd())
    u.savefig(figure=fig, file_name=Path('figs') / 'rnd_amplitude_vs_rnd_phase_vs_real.pdf',
              transparent=True)

    # u.play(np.concatenate((np.zeros(int(fs * 0.5)), y_rnd_amplitude, np.zeros(int(fs * 0.5)))), fs=fs, volume=0.4)
    # u.play(np.concatenate((np.zeros(int(fs * 0.5)), y_rnd_phase, np.zeros(int(fs * 0.5)))), fs=fs, volume=0.4)
    # u.play(np.concatenate((np.zeros(int(fs * 0.5)), y_real, np.zeros(int(fs * 0.5)))), fs=fs, volume=0.4)
    # u.play(np.concatenate((np.zeros(int(fs * 0.5)), y_real_original, np.zeros(int(fs * 0.5)))), fs=fs, volume=0.4)
