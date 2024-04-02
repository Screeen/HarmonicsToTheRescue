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

f_cutoff = 600
fs = 48000
# f0 = 100
Nw = 4096
# N = int(np.round(0.11 * fs))  # approximate N
N = int(np.round(0.17 * fs))
N = int(np.round(0.24 * fs))
L = int(np.ceil(N / Nw))
N = L * Nw  # adjust N to be a multiple of Nw
R_shift_samples = Nw // 3
snr = np.inf
num_realizations = 100

module_parent = Path(__file__).resolve().parent
dataset_path_parent = module_parent.parent / 'datasets' / 'north_texas_vowels' / 'data'
file_name = 'kadpal03.wav'
if dataset_path_parent.exists():
    pp = dataset_path_parent / file_name
else:
    warnings.warn(f"Path {dataset_path_parent} does not exist. Use sample file.")
    pp = module_parent / 'audio' / file_name

y_real = u.load_audio_file(pp, fs_=fs, offset_seconds=0.045, N_num_samples=N, smoothing_window=False)
f0_stats, _ = manager.Manager.find_f0_from_recording(y_real, fs, 512, 2048)
f0 = np.round(f0_stats[1])
num_harmonics = (f_cutoff - f0) // f0
print(f"{f0 = }, {num_harmonics = }")

# Post-process real signal (low-pass filter, add noise, normalize)
sos = scipy.signal.butter(10, f_cutoff, 'low', fs=fs, output='sos')
y_real = scipy.signal.sosfilt(sos, y_real)
y_real = manager.Manager.add_noise_snr(y_real, snr_db=snr)[0]
y_real = helper.normalize(y_real)

# Generate synthetic signals: random amplitude and random phase processes
sin_gen = helper.SinusoidGenerator()
y_rnd_amplitude_list = []
y_rnd_phase_list = []
for ii in range(num_realizations):
    y_a, y_p = sin_gen.generate_two_harmonic_processes(freqs_hz=f0 * np.arange(num_harmonics + 1),
                                                       Nw_win_length=Nw, L_num_frames=L,
                                                       normalize_flag=False, fs=fs)
    y_a, _ = manager.Manager.add_noise_snr(y_a, snr_db=snr)
    y_p, _ = manager.Manager.add_noise_snr(y_p, snr_db=snr)

    y_a = helper.normalize(y_a)
    y_p = helper.normalize(y_p)

    y_rnd_amplitude_list.append(y_a)
    y_rnd_phase_list.append(y_p)

# SCF plots
if show_combined_scf_plots:
    # Calculate the spectral correlation functions
    dft_props = {'nfft': Nw, 'noverlap': Nw - R_shift_samples, 'window': 'hann', 'nw': Nw, 'fs': fs}
    scf_cfg = {'dft_props': dft_props, 'num_harmonics': num_harmonics, 'alpha_max_hz': f_cutoff + f0 // 2}
    plot_cfg = {'amp_range': (-70, 0), 'figsize': (7, 3.5), 'show_figures': show_single_scf_plots}

    h = helper.Helper()
    scf_dict_first_realization_amp, scf_dict_avg_amp, sig_prop_amp = h.compute_2d_scf(y_rnd_amplitude_list, dft_props,
                                                                                      scf_cfg, f0)
    scf_dict_first_realization_phase, scf_dict_avg_phase, sig_prop_phase = h.compute_2d_scf(y_rnd_phase_list, dft_props,
                                                                                            scf_cfg, f0)
    scf_dict_real, _, sig_prop_real = h.compute_2d_scf([y_real], dft_props, scf_cfg, f0)

    f2ds_real, x_ticks_labels, y_ticks_labels = h.plot_2d_scf(scf_dict_real, sig_prop_real, **plot_cfg, title='Real',
                                                              return_tick_labels=True)
    f2ds_phase = h.plot_2d_scf(scf_dict_first_realization_phase, sig_prop_phase, **plot_cfg, title='Phase (single)')
    f2ds_phase_avg = h.plot_2d_scf(scf_dict_avg_phase, sig_prop_phase, **plot_cfg, title='Phase (average)')
    f2ds_amp_s = h.plot_2d_scf(scf_dict_first_realization_amp, sig_prop_amp, **plot_cfg, title='Amplitude (single)')
    f2ds_amp_avg = h.plot_2d_scf(scf_dict_avg_amp, sig_prop_amp, **plot_cfg, title='Amplitude (average)')

    # Combine the SCFs plots into a single figure
    import matplotlib.pyplot as plt

    # Assume existing_figs is a list of existing figures
    existing_figs = [*f2ds_real, None, *f2ds_phase, *f2ds_phase_avg, *f2ds_amp_s, *f2ds_amp_avg,]
    # titles = ['Real', None, 'Amplitude (single)', 'Amplitude (average)', 'Phase (single)', 'Phase (average)']
    titles = ['Voiced speech', None,
              'WSS harmonic model', 'WSS harmonic model (avg)',
              'Cyclostationary harmonic model', 'Cyclostationary harmonic model (avg)']

    # Create a new figure
    fig = plt.figure(figsize=(6, 4.5), layout='compressed')

    # Loop over the list of existing figures
    # The new figure will have 3 rows and 2 columns. The first row will contain the real signal and an empty spot,
    # the second row will contain the random phase and random amplitude signals, and the third row will contain the
    # average SCFs.
    for i, (existing_fig, title) in enumerate(zip(existing_figs, titles)):
        ax = fig.add_subplot(3, 2, i + 1)
        im = h.create_subplot(existing_fig, title, ax, xy_ticks=(x_ticks_labels, y_ticks_labels))

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
    u.savefig(figure=fig, file_name=PosixPath('figures_scf') / '2d_scfs_real_vs_rnd_amp_vs_rnd_phase.pdf')

# Time-domain plots
if show_combined_time_domain_plots:
    y_rnd_amplitude = y_rnd_amplitude_list[0]
    y_rnd_phase = y_rnd_phase_list[0]

    signals = [y_real, y_rnd_phase, y_rnd_amplitude]
    titles = [
        'Voiced speech $s_{\\text{real}}(n)$',
        'WSS harmonic model $s_{\\text{ph}}(n)$',
        'Cyclostationary harmonic model $s_{\\text{amp}}(n)$',
    ]
    # '$\check{s}_1(n; \mathbf{\Phi})$',
    # '$\check{s}_2(n; \mathbf{b})$'

    fig = u.plot(signals, titles=titles, fs=fs)
    axes = fig.get_axes()
    for ax in axes:
        ax.set_ylim(-1, 1)

    # On the second ax (random phase), insert a vertical red line every 2048 samples
    for i in range(0, len(y_rnd_phase), Nw):
        axes[1].axvline(i, color='r', linestyle='--', linewidth=1)
    # fig.set_size_inches(4.5, 3.8)
    fig.show()
    u.check_create_folder(folder_name='figures_waveform', parent_folder=Path.cwd())
    u.savefig(figure=fig, file_name=Path('figures_waveform') / 'rnd_amplitude_vs_rnd_phase_vs_real.pdf')

    # u.play(np.concatenate((np.zeros(int(fs * 0.5)), y_rnd_amplitude, np.zeros(int(fs * 0.5)))), fs=fs, volume=0.4)
    # u.play(np.concatenate((np.zeros(int(fs * 0.5)), y_rnd_phase, np.zeros(int(fs * 0.5)))), fs=fs, volume=0.4)
    # u.play(np.concatenate((np.zeros(int(fs * 0.5)), y_real, np.zeros(int(fs * 0.5)))), fs=fs, volume=0.4)
    # u.play(np.concatenate((np.zeros(int(fs * 0.5)), y_real_original, np.zeros(int(fs * 0.5)))), fs=fs, volume=0.4)
