import copy

import manager
import spectral_correlation_estimator as sce
import plotter
import utils as u

import numpy as np

eps = 1e-10
def normalize(x):
    x = x / (eps + np.max(np.abs(x)))
    x = x / 1.1
    return x


class SinusoidGenerator:
    def __init__(self):
        pass

    @staticmethod
    def generate_two_harmonic_processes(freqs_hz, Nw_win_length, L_num_frames, fs=1, amplitude_single_harmonic=0.5):
        """
        Generate two harmonic processes.
        The first process has Nw_win_length*L_num_frames samples. Per each harmonic (sinusoid),
        the amplitude is a AR(1) process filtered by a moving average filter, and the phase is a uniform random variable.
        The second process has the same number of samples, but the amplitudes and the phases are calculated independently
        for each of the L_num_frames.
        """

        # First process: fixed phases, WSS amplitudes
        # It is a cyclostationary process
        phases = np.linspace(-np.pi / 4, np.pi, len(freqs_hz), endpoint=False)
        sin_rnd_amplitude = u.generate_harmonic_process(freqs_hz, Nw_win_length * L_num_frames, fs=fs,
                                                        phases=phases, amp_harmonic=amplitude_single_harmonic)

        # Second process: concatenation of L_num_frames independent processes
        # Each process is WSS: has fixed amplitudes and random phases
        short_sins_rnd_phase = []
        amplitudes = np.ones((len(freqs_hz), Nw_win_length)) * 0.5
        for ll in range(L_num_frames):
            short_sins_rnd_phase.append(
                u.generate_harmonic_process(freqs_hz, Nw_win_length, fs=fs, amplitudes_over_time=amplitudes))
        sin_rnd_phase = np.concatenate(short_sins_rnd_phase)

        return sin_rnd_amplitude, sin_rnd_phase

    @staticmethod
    def ar_process(N_, variance_=1., p=10, mean=0.):
        # Set the AR(p) parameters randomly
        ar = u.rng.uniform(0, 1, p)
        x = np.zeros(N_)
        for jj in range(p, N_):
            x[jj] = mean + np.sum(ar * x[jj - p:jj]) + u.rng.normal(0, variance_)
        return x


class Helper:
    def __init__(self):
        pass

    @staticmethod
    def prepare_parameters_cyclic_spectrum_estimation(dft_props, num_harmonics, alpha_max_hz=1000, f0=None,
                                                      conjugate_scf=False, N_num_samples_=-1):

        # Prepare parameters for 2D spectral correlation function.
        # The input is a list of sample paths of the signal. The output is a dictionary of parameters for the spectral
        # correlation function and a dictionary of parameters for the signal.

        # Read parameters
        Nw = dft_props['nfft']
        fs = dft_props['fs']

        # Set parameters
        names_scf_estimators = ['acp']
        R_shift_samples = Nw // 3

        L_num_frames = np.ceil(1 + (N_num_samples_ - Nw) / R_shift_samples).astype(int)

        alphas_dirichlet = sce.SpectralCorrelationEstimator.get_alpha_vec_hz_dirichlet(L_num_frames,
                                                                                       R_shift_samples_=R_shift_samples, fs_=fs)
        m = manager.Manager()
        delta_f, delta_alpha_dict = m.compute_spectral_and_cyclic_resolutions(fs, Nw, names_scf_estimators,
                                                                              alphas_dirichlet, N_num_samples_)

        # delta_alpha_dict['acp'] = fs / Nw
        # print(f"Cyclic resolution set to {delta_alpha_dict['acp'] = :.2f} Hz instead of "
        #       f"delta_alpha_min = {fs / (L_num_frames * R_shift_samples):.2f} Hz.")

        sig_prop = {'alpha_min_hz': 0, 'alpha_max_hz': alpha_max_hz,
                    # 'delta_alpha_dict': delta_alpha_dict,
                    'simulated_signal': True}

        if f0 is not None:
            sig_prop['f0_range'] = (f0 - 1, f0, f0 + 1)
            sig_prop['f0_over_time'] = f0

            # Maximum spectral frequency shown in the plot
            sig_prop['f_max_bin'] = int(np.ceil((((f0 + 1) * (num_harmonics + 0.5)) / (fs / 2)) * (dft_props['nfft'] / 2)))
            sig_prop['f_max_hz'] = sig_prop['f_max_bin'] * delta_f

        sce_prop = {'names_scf_estimators': names_scf_estimators, 'dft_props': dft_props,
                    'alpha_min_hz': sig_prop['alpha_min_hz'], 'alpha_max_hz': sig_prop['alpha_max_hz'],
                    'delta_alpha_dict': delta_alpha_dict, 'normalize_scf_to_1': True, 'coherence': False,
                    'conjugate_scf': conjugate_scf}

        return sce_prop, sig_prop

    @classmethod
    def compute_cyclic_spectra_all_realizations(cls, sample_paths_list, dft_props, scf_cfg, f0):
        # Evaluate 2d spectral correlation function both from each realization of the signal (in y_sig_samples_list)
        # SCFs = spectral correlation functions

        SCE = sce.SpectralCorrelationEstimator(dft_props=dft_props)
        sce_props, sig_props = cls.prepare_parameters_cyclic_spectrum_estimation(N_num_samples_=sample_paths_list[0].shape[-1],
                                                                                 f0=f0, **scf_cfg)

        # delta_alpha_dict['acp'] = fs / Nw
        # print(f"Cyclic resolution set to {delta_alpha_dict['acp'] = :.2f} Hz instead of "
        #       f"delta_alpha_min = {fs / (L_num_frames * R_shift_samples):.2f} Hz.")
        # realizations_or_frames = np.maximum(dft_props)
        # sce_props['delta_alpha_dict']['acp'] = dft_props['fs'] / (dft_props['R_shift_samples'] * realizations_or_frames)

        estimated_scfs_and_freqs_list = []
        for yy in sample_paths_list:
            cyc_spectrum_dict_single_realization = SCE.run_spectral_correlation_estimators(x=yy, **sce_props)
            estimated_scfs_and_freqs_list.append(cyc_spectrum_dict_single_realization)

        # scf_dict_first_path = estimated_scfs_and_freqs_list[0]
        # scf_estimator_name = sce_props['names_scf_estimators'][0]

        # scf_dict_avg = copy.deepcopy(scf_dict_first_path)
        # scf_dict_avg[scf_estimator_name]['scf'] = np.mean(
        #     np.array([scf_dict[scf_estimator_name]['scf'] for scf_dict in estimated_scfs_and_freqs_list]), axis=0)

        # return scf_dict_first_path, scf_dict_avg, sig_props
        return *estimated_scfs_and_freqs_list, sig_props

    @staticmethod
    def plot_2d_scf(scf_dict, sig_prop, amp_range, figsize=None, title=None, show_figures=True, return_tick_labels=False):

        pl = plotter.Plotter(which_plots={'2d': True}, amp_range=amp_range, show_figures=show_figures)
        pl.update_data(sig_props=sig_prop)
        scf_fig, xlabels, ylabels = pl.plot_2d_scf(scf_dict, figsize=figsize, title=title)

        if return_tick_labels:
            return scf_fig, xlabels, ylabels
        else:
            return scf_fig
