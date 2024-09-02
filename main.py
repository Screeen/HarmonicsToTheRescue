import warnings
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import plotter
import spectral_correlation_estimator as sce
import manager as si_manager
import system_identifier as si
import utils as u

u.set_printoptions_numpy()
u.set_plot_options(use_tex=True)

m = si_manager.Manager()
H_hat = None
scf_xx = None
alpha_min_hz = 0

plot_area_size = 3
plt_config = {'font_size': 'medium',
              'xscale_log': False,
              'legend_num_cols': 1,
              'legend_font_size': 'medium',
              # 'ylim': (0.0, 0.35),
              'ylim': (0.0, 1.0),
              'title': '',
              'metric_name': 'RMSE'}

fs = 16000
# variation_name = 'nfft'
# variation_names = ['nfft', 'snr']
variation_names = ['snr']
# variation_names = ['nfft']

# N_num_samples_sim = int(0.25 * fs)
# N_num_samples_real = int(0.25 * fs)
offset_load_real = 0.08
selected_people = ['1', '2']  # 1 is only male, 1,2 is male and female
alpha_max_hz = 4000
f_max_hz = alpha_max_hz

# desired_snrs_config = [-20, -10]
desired_snrs_config = [-20, -10, 0, 10, 20]
D_length_seconds_config = [0.1, 0.2, 0.3, 0.4]  # seconds
f0_config = [25, 50, 100, 150, 200, 300]
# nfft_config = [128, 256]
nfft_config = [128, 256, 512, 1024, 2048]
# N_num_samples_config = [0.5]  # seconds
# desired_snrs_config = [30]
configs_dict = {'snr': desired_snrs_config, 'duration': D_length_seconds_config, 'f0': f0_config, 'nfft': nfft_config}

simulated_signal = True
round_f0_sim_delta_f = False
round_f0_sim_delta_alpha = False

nfft_default = 256
snr_db_default = 0
N_num_samples_default = int(0.25 * fs)

num_montecarlo = 10

compute_coherence = True
filtered_noise = False
if filtered_noise:
    warnings.warn("LP filtered noise is added to the signal")

which_plots = {
    'time_psd': False,
    '1d': False,
    '2d': False,
    '3d': False,
    'estimated_h_time': False,
    'estimated_h_freq': False,
    'f0_spectrogram': False,  # if true, plot spectrogram and f0 and exit
    'error_vs_variation_param': True
}

save_figures = False
show_figures = True
# amp_range_plot = (-180, 0)
amp_range_plot = (None, None)

# names_scf_estimators = ['sample_cov', 'acp']
names_scf_estimators = ['acp']
# names_scf_estimators = ['dirichlet']
# names_scf_estimators = ['acp', 'sample_cov', 'dirichlet']
# names_scf_estimators = []
# names_scf_estimators = ['sample_cov', 'dirichlet', 'acp', 'psd']
indices_scf_estimators = [names_scf_estimators.index(name) for name in names_scf_estimators]

# names_h_estimators = ['Antoni', 'Wiener', 'Time-domain Wiener']
names_h_estimators = ['Antoni', 'Wiener']
# names_h_estimators = ['Wiener']
# names_h_estimators = []
indices_h_estimators = [names_h_estimators.index(name) for name in names_h_estimators]

# With these parameters ACP and Dirichlet estimators have same cyclic frequency resolution
# R_shift_samples = np.ceil(fs / (2 * alpha_max_hz)).astype(int)
# L_num_frames = np.ceil(1 + np.floor((N_num_samples - Nw_nfft) / R_shift_samples)).astype(int)

plotter_obj = plotter.Plotter(which_plots=which_plots, save_figures=save_figures,
                              show_figures=show_figures, amp_range=amp_range_plot,
                              names_scf_estimators=names_scf_estimators, names_h_estimators=names_h_estimators,
                              indices_h_estimators=indices_h_estimators)

results_list = []
for variation_name in variation_names:
    variations_list, variations_list_display = (
        m.define_variation_list(variation_name, configs_dict, simulated_signal, fs))

    num_h_estimators = len(names_h_estimators)
    num_scf_estimators = len(names_scf_estimators)
    num_variations = len(variations_list)
    rmse_all_realizations = np.zeros((num_montecarlo, num_variations, num_h_estimators, num_scf_estimators))
    rmse_mean = np.zeros((num_variations, num_h_estimators, num_scf_estimators))
    rmse_conf_int = np.zeros((num_variations, num_h_estimators, num_scf_estimators))
    # errors_array = np.zeros((num_variations, num_h_estimators, num_scf_estimators, 3))

    for variation_idx, variation_parameter in enumerate(variations_list):

        desired_snr = variation_parameter if variation_name == 'snr' else snr_db_default
        f0_sim = variation_parameter if variation_name == 'f0' else None
        nfft = variation_parameter if variation_name == 'nfft' else nfft_default
        N_num_samples = variation_parameter if variation_name == 'duration' else N_num_samples_default
        print(f"{variation_name} = {variation_parameter}")

        nw = nfft
        R_shift_samples = nw // 3
        noverlap = nw - R_shift_samples
        # num_samples_h = nfft // 2
        num_samples_h = nfft

        assert N_num_samples >= nw, f"Signal length {N_num_samples} is shorter than one frame {nw}"

        dft_properties = {'nfft': nfft, 'nw': nw, 'fs': fs, 'noverlap': noverlap}
        print(f"{dft_properties['nfft'] = }, {dft_properties['nw'] = }")
        SCE = sce.SpectralCorrelationEstimator(dft_properties)
        L_num_frames = np.ceil(1 + (N_num_samples - nw) / R_shift_samples).astype(int)

        # alpha_max_hz = np.ceil(num_harmonics * (np.ceil(f0_sim))) if simulated_signal else np.ceil(num_harmonics * 150)
        # alpha_max_hz = np.ceil(num_harmonics * (np.ceil(f0_sim))) if simulated_signal else 4000

        alphas_dirichlet = sce.SpectralCorrelationEstimator.get_alpha_vec_hz_dirichlet(L_num_frames, R_shift_samples,
                                                                                       fs, alpha_max_hz)
        if not alphas_dirichlet.any():
            warnings.warn("Alpha vector is invalid. Use same resolution as DFT")
            alphas_dirichlet = np.fft.fftfreq(nfft, 1 / fs)
        delta_f, delta_alpha_dict = m.compute_spectral_and_cyclic_resolutions(fs, nfft, names_scf_estimators,
                                                                              alphas_dirichlet, N_num_samples)

        sig_prop = {
            # 'num_harmonics': num_harmonics,
            'f_max_hz': f_max_hz,
            'f_max_bin': int(np.ceil(f_max_hz / delta_f)),
            'simulated_signal': simulated_signal, 'alpha_min_hz': alpha_min_hz,
            'N_num_samples': N_num_samples, 'alpha_max_hz': alpha_max_hz, 'delta_alpha_dict': delta_alpha_dict,
            'delta_f': delta_f, 'snr': desired_snr}

        sce_dict = {'names_scf_estimators': names_scf_estimators, 'dft_props': dft_properties,
                    'alpha_min_hz': sig_prop['alpha_min_hz'], 'alpha_max_hz': sig_prop['alpha_max_hz'],
                    'delta_alpha_dict': sig_prop['delta_alpha_dict'], 'normalize_scf_to_1': False,
                    'coherence': compute_coherence, }

        for ii in range(num_montecarlo):

            # h(n) is a random filter with exponentially decaying envelope.
            h = m.generate_impulse_response(num_samples_h=num_samples_h)
            H = np.fft.rfft(h, n=nfft)

            if simulated_signal:
                if f0_sim is None:
                    f0_sim = u.rng.integers(90, 250)

                if round_f0_sim_delta_f:
                    f0_sim = np.ceil(f0_sim / delta_f) * delta_f

                sig_prop['f0_range'] = (f0_sim - 1, f0_sim, f0_sim + 1)
                sig_prop['f0_over_time'] = np.ones(N_num_samples) * f0_sim
                s_in = m.generate_simulated_signal(sig_prop, fs=dft_properties['fs'])

            else:
                s_in = m.load_vowel_recording(N_num_samples, fs, offset_=offset_load_real,
                                              selected_people=selected_people)
                s_in = m.remove_mean_normalize(s_in)

                sig_prop['f0_range'], sig_prop['f0_over_time'] = m.find_f0_from_recording(s_in, fs, R_shift_samples,
                                                                                          nfft)
                if 0 in sig_prop['f0_range']:
                    continue

            sig_prop['num_harmonics'] = int(np.ceil(sig_prop['f_max_hz'] / sig_prop['f0_range'][1]))

            # Calculate spectral and cyclic frequencies (bins) which fall on harmonics of f0
            spectral_bins_harmonics = m.calculate_harmonic_frequencies(sig_prop['f0_range'], sig_prop['num_harmonics'],
                                                                       sig_prop['delta_f'], sig_prop['f_max_hz'] // 2)

            cyclic_bins_harmonics_dict = dict.fromkeys(sig_prop['delta_alpha_dict'].keys())
            for scf_est_name, delta_alpha in sig_prop['delta_alpha_dict'].items():
                cyclic_bins_harmonics_dict[scf_est_name] = m.calculate_harmonic_frequencies(sig_prop['f0_range'],
                                                                                            sig_prop['num_harmonics'],
                                                                                            delta_alpha,
                                                                                            sig_prop['alpha_max_hz'])
            # Convolve clean signal with random LTI system h(n).
            d_out, D = m.convolution_with_system(h=h, s=s_in, **dft_properties)

            SI = si.SystemIdentifier(sig_props=sig_prop, spectral_bins_harmonics=spectral_bins_harmonics,
                                     cyclic_bins_harmonics_dict=cyclic_bins_harmonics_dict, H_true=H)

            evaluated_bins, d_psd = m.choose_evaluated_bins(d_out, dft_properties, spectral_bins_harmonics)

            # Synthesize noisy signal
            noise_floor = 40  # dB
            z_in, n_in = m.add_noise_snr(s_in, desired_snr, fs, lp_filtered_noise=filtered_noise)  # input noise
            x_out, n_out = m.add_noise_snr(d_out, noise_floor, fs, lp_filtered_noise=filtered_noise)  # output noise

            # u.plot([s, d, z, x], fs=fs, subplot_height=1.5,
            #    titles=['clean input s', 'clean output d', 'noisy input z', 'noisy output x'],)

            # Estimate spectral correlation functions (2D Fourier of cross-correlation functions R_xy(t, tau))
            scf_zz = SCE.run_spectral_correlation_estimators(x=z_in, **sce_dict)
            scf_xz = SCE.run_spectral_correlation_estimators(x=x_out, y=z_in, **sce_dict)
            # scf_xx = SCE.run_spectral_correlation_estimators(x=x_out, **sce_dict)
            # scf_ss = SCE.run_spectral_correlation_estimators(x=s_in, **sce_dict)

            # Estimate transfer function if at least one SCF estimator is used
            H_hat = SI.run(S_in_in_dict=scf_zz, S_out_in_dict=scf_xz,
                           nfft=dft_properties['nfft'], names_h_estimators=names_h_estimators, x=z_in, y=x_out,
                           names_scf_estimators=names_scf_estimators, S_out_out_dict=scf_xx)

            # Plot time, PSD, SCF, and estimated transfer function
            if ii == 0:
                system_properties = {'h': h, 'H': H, 'H_hat': H_hat}
                plotter_obj.update_data(dft_props=dft_properties, sig_props=sig_prop, sys_props=system_properties,
                                        ref_sig=d_out, ref_sig_psd=d_psd, evaluated_bins=evaluated_bins)

                _, s_psd = m.choose_evaluated_bins(s_in, dft_properties, spectral_bins_harmonics)
                plotter_obj.update_data(ref_sig=s_in, ref_sig_psd=s_psd)
                plotter_obj.plot_all(spectral_correlation_functions=scf_xz, f0=sig_prop['f0_range'][1])

            # Calculate Root Mean Squared Error (RMSE) for each realization
            for h_est_idx in indices_h_estimators:
                for scf_est_idx in indices_scf_estimators:
                    rmse_all_realizations[ii, variation_idx, h_est_idx, scf_est_idx] = (
                        m.calculate_rmse(H[evaluated_bins], H_hat[evaluated_bins][:, h_est_idx, scf_est_idx]))

        # Calculate Root Mean Squared Error (RMSE) and 95% confidence interval
        # Three outputs because we measure errors on 3 different quantities: abs, real, imag
        conf_int_factor = 1.96 / np.sqrt(num_montecarlo)
        for h_est_idx in indices_h_estimators:
            for scf_est_idx in indices_scf_estimators:
                rmse_ii = rmse_all_realizations[:, variation_idx, h_est_idx, scf_est_idx]
                rmse_mean[variation_idx, h_est_idx, scf_est_idx] = np.mean(rmse_ii, axis=0)
                rmse_conf_int[variation_idx, h_est_idx, scf_est_idx] = np.std(rmse_ii, axis=0) * conf_int_factor

        # end of variation loop

    errors_array = np.stack((rmse_mean, rmse_mean - rmse_conf_int, rmse_mean + rmse_conf_int), axis=-1)
    errors_array, names_h_estimators_display = m.prepare_errors_array_for_plot(errors_array,
                                                                               names_h_estimators,
                                                                               names_scf_estimators,
                                                                               indices_h_estimators,
                                                                               indices_scf_estimators)

    # Quick and dirty fix to have cleaner names in the plot
    names_h_estimators_display_new = []
    for name in names_h_estimators_display:
        if 'antoni' in name.lower():
            names_h_estimators_display_new.append('Antoni (CS model)')
        elif 'wiener' in name.lower():
            names_h_estimators_display_new.append('Wiener (WSS model)')
        elif 'gardner' in name.lower():
            names_h_estimators_display_new.append('Gardner')

    results_dict = {'errors_array': errors_array, 'x_values': variations_list_display,
                    'algo_names': names_h_estimators_display_new, 'x_label': variation_name.upper()}
    results_list.append(results_dict)

# Plot results for each variation_name
num_plots = len(variation_names)
height = 1.78
# height = 2.0 if not simulated_signal else 1.78
fig, axes = plt.subplots(ncols=num_plots, figsize=(1 + num_plots * plot_area_size, height), layout='compressed',
                         sharey=True, squeeze=False)
for idx, (ax, results_dict) in enumerate(zip(axes.flat, results_list)):
    pos = 'best' if simulated_signal else 'none'
    # pos = 'none'  # alternatives: 'outside_plot', 'none', 'best'
    # if idx == len(results_list) - 1 and simulated_signal:
    #     pos = 'outside_plot'
    plotter_obj.plot_errors(fig_ax_tuple=(fig, ax), legend_positioning=pos, **results_dict, **plt_config)

suptitle = 'Simulated data' if simulated_signal else 'Real data'
if simulated_signal:
    fig.suptitle(suptitle, fontsize='xx-large', y=1.23)
else:
    fig.suptitle(suptitle, fontsize='xx-large')
fig.show()
u.check_create_folder('figures_errors')
u.savefig(fig, Path('figures_errors') / f'errors_{"_".join(suptitle.lower().split(" "))}.pdf',
          transparent=True)
