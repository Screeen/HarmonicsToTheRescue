import librosa.display
import matplotlib.pyplot as plt
import warnings
from pathlib import PosixPath, Path
import numpy as np
import scipy

import manager
import utils as u
import zz_plot_real_synthetic_vowel_helper as helper

u.set_plot_options(use_tex=True)
u.set_printoptions_numpy()

show_single_scf_plots = False
show_combined_scf_plots = True
show_combined_time_domain_plots = True

f_cutoff = 1000
fs = 16000
# f0 = 100
Nw = 1024
# N = int(np.round(0.11 * fs))  # approximate N
# N = int(np.round(0.17 * fs))
N = int(np.round(0.24 * fs))
N = int(np.round(100 * fs))
L = int(np.ceil(N / Nw))
N = L * Nw  # adjust N to be a multiple of Nw
R_shift_samples = Nw // 3
snr = np.inf

module_parent = Path(__file__).resolve().parent
dataset_path_parent = module_parent.parent / 'datasets' / 'Anechoic'
file_name = 'SI Harvard Word Lists Female.wav'
if dataset_path_parent.exists():
    pp = dataset_path_parent / file_name
else:
    warnings.warn(f"Path {dataset_path_parent} does not exist. Use sample file.")
    pp = module_parent / 'audio' / file_name

y_real_original = u.load_audio_file(pp, fs_=fs, offset_seconds=0.045, N_num_samples=N, smoothing_window=False)

# Select first 10s only
offset = int(28.75 * fs)
y_real = y_real_original[offset: offset + int(.25 * fs)]

# Plot spectrogram with librosa
# S = np.abs(librosa.stft(y_real, n_fft=Nw, hop_length=Nw // 3))
# plt.figure(figsize=(6, 3))
# librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
# plt.tight_layout()
# plt.show()
# exit()

f0_stats, _ = manager.Manager.find_f0_from_recording(y_real, fs, 512, 2048)
f0 = np.round(f0_stats[1])
num_harmonics = (f_cutoff - f0) // f0
print(f"{f0 = }, {num_harmonics = }")

# Post-process real signal (low-pass filter, add noise, normalize)
sos = scipy.signal.butter(10, f_cutoff, 'low', fs=fs, output='sos')
y_real = scipy.signal.sosfilt(sos, y_real)
y_real = manager.Manager.add_noise_snr(y_real, snr_db=snr)[0]
y_real = helper.normalize(y_real)

# SCF plots
if show_combined_scf_plots:
    # Calculate the spectral correlation functions
    dft_props = {'nfft': Nw, 'noverlap': Nw - R_shift_samples, 'window': 'hann', 'nw': Nw, 'fs': fs}
    scf_cfg = {'dft_props': dft_props, 'num_harmonics': num_harmonics, 'alpha_max_hz': f_cutoff + f0 // 2,
               'conjugate_scf': True}
    plot_cfg = {'amp_range': (-70, 0), 'figsize': (7, 3.5), 'show_figures': show_single_scf_plots}

    h = helper.Helper()
    scf_dict_real, _, sig_prop_real = h.compute_cyclic_spectra_all_realizations([y_real], dft_props, scf_cfg, f0)

    f2ds_real, x_ticks_labels, y_ticks_labels = h.plot_2d_scf(scf_dict_real, sig_prop_real, **plot_cfg, title='Real',
                                                              return_tick_labels=True)

    # Combine the SCFs plots into a single figure
    import matplotlib.pyplot as plt

    # Assume existing_figs is a list of existing figures
    existing_figs = [*f2ds_real, None]
    titles = ['Voiced speech', None]

    # Create a new figure
    fig = plt.figure(figsize=(6, 4.5), layout='compressed')

    # Loop over the list of existing figures
    # The new figure will have 3 rows and 2 columns. The first row will contain the real signal and an empty spot,
    # the second row will contain the random phase and random amplitude signals, and the third row will contain the
    # average SCFs.
    for i, (existing_fig, title) in enumerate(zip(existing_figs, titles)):
        ax = fig.add_subplot(1, 2, i + 1)
        im = h.fig_to_subplot(existing_fig, title, ax, xy_ticks=(x_ticks_labels, y_ticks_labels))

    # Hide top right subplot
    fig.get_axes()[1].axis('off')

    # The colorbar will be shared among all subplots and should be placed in the first row to the right,
    # in place of the empty spot. So it should be shifted to the left
    # Label is 'Normalize magnitude [dB]'
    fig.colorbar(im, ax=fig.get_axes()[1], orientation='horizontal', pad=-0.75, label='Normalized magnitude [dB]')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in fig.get_axes():
        ax.label_outer()

    fig.show()
    u.check_create_folder(folder_name='figures_scf', parent_folder=Path.cwd())
    # u.savefig(figure=fig, file_name=PosixPath('figures_scf') / '2d_scfs_real_vs_rnd_amp_vs_rnd_phase.pdf')

