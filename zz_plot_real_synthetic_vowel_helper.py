import copy

import spectral_correlation_estimator as sce
import plotter
import manager
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
    def generate_two_harmonic_processes(freqs_hz, Nw_win_length, L_num_frames, normalize_flag=True, fs=1):
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
                                                        phases=phases)

        # Second process: concatenation of L_num_frames independent processes
        # Each process is WSS: has fixed amplitudes and random phases
        short_sins_rnd_phase = []
        amplitudes = np.ones((len(freqs_hz), Nw_win_length))
        for ll in range(L_num_frames):
            short_sins_rnd_phase.append(
                u.generate_harmonic_process(freqs_hz, Nw_win_length, fs=fs, amplitudes_over_time=amplitudes))
        sin_rnd_phase = np.concatenate(short_sins_rnd_phase)

        if normalize_flag:
            sin_rnd_amplitude = normalize(sin_rnd_amplitude)
            sin_rnd_phase = normalize(sin_rnd_phase)

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
    def create_subplot(existing_fig, title, ax, xy_ticks):

        if existing_fig is None:
            return None

        # Retrieve the image data from the existing figure
        img = existing_fig.axes[0].collections[0].get_array().data

        # Retrieve vmin and vmax from the existing figure
        vmin, vmax = existing_fig.axes[0].collections[0].get_clim()

        # Retrieve the x ticks, y ticks, color map, and labels from the existing figure
        # x_locator = existing_fig.axes[0].xaxis.get_major_locator()
        # y_locator = existing_fig.axes[0].yaxis.get_major_locator()
        # x_minor_locator = existing_fig.axes[0].xaxis.get_minor_locator()
        # y_minor_locator = existing_fig.axes[0].yaxis.get_minor_locator()
        cmap = existing_fig.axes[0].collections[0].get_cmap()
        # xlabel = existing_fig.axes[0].get_xlabel()
        # ylabel = existing_fig.axes[0].get_ylabel()
        # xticklabels = [label.get_text() for label in existing_fig.axes[0].get_xticklabels()]
        # yticklabels = [label.get_text() for label in existing_fig.axes[0].get_yticklabels()]

        # Display the image data in the new subplot
        im = ax.pcolormesh(*xy_ticks, img, antialiased=True, vmin=vmin, vmax=vmax, cmap=cmap)

        # xlabel = 'Spectral freq. $\\alpha_p~f_s/2\pi$ [kHz]'
        # ylabel = '$\\omega_k~f_s/2\pi$ [kHz]'
        ylabel = 'Freq.~$\\omega_k$ [kHz]'
        xlabel = 'Cyclic freq.~$\\alpha_p$ [kHz]'

        # Set the title of the subplot
        ax.set_title(title)

        # Apply these properties to the new subplot
        # ax.xaxis.set_major_locator(x_locator)
        # ax.yaxis.set_major_locator(y_locator)
        # ax.xaxis.set_minor_locator(x_minor_locator)
        # ax.yaxis.set_minor_locator(y_minor_locator)
        # ax.set_xticklabels(xticklabels)
        # ax.set_yticklabels(yticklabels)
        # im.set_cmap(cmap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return im

    @staticmethod
    def prepare_parameters_2d_scf(y_sig_samples_list, dft_props, num_harmonics, alpha_max_hz = 1000, f0=None):
        # Prepare parameters for 2D spectral correlation function.
        # The input is a list of sample paths of the signal. The output is a dictionary of parameters for the spectral
        # correlation function and a dictionary of parameters for the signal.

        if not isinstance(y_sig_samples_list, list):
            y_sig_samples_list = [y_sig_samples_list]

        # Read parameters
        Nw = dft_props['nfft']
        fs = dft_props['fs']
        N_num_samples = len(y_sig_samples_list[0])

        # Set parameters
        names_scf_estimators = ['dirichlet']
        R_shift_samples = Nw // 3

        L_num_frames = np.ceil(1 + (N_num_samples - Nw) / R_shift_samples).astype(int)

        alphas_dirichlet = sce.SpectralCorrelationEstimator.get_alpha_vec_hz_dirichlet(L_num_frames, R_shift_samples_=R_shift_samples, fs_=fs)
        m = sys_identifier_manager.Manager()
        delta_f, delta_alpha_dict = m.compute_spectral_and_cyclic_resolutions(fs, Nw, names_scf_estimators,
                                                                              alphas_dirichlet, N_num_samples)

        sig_prop = {'alpha_min_hz': 0, 'alpha_max_hz': alpha_max_hz, 'delta_alpha_dict': delta_alpha_dict,
                    'simulated_signal': True}
        sig_prop_real = m.estimate_f0_generate_props_real(y_sig_samples_list[0], fs, dft_props['nfft'], R_shift_samples)
        sig_prop.update(sig_prop_real)

        if f0 is not None:
            sig_prop['f0_range'] = (f0 - 1, f0, f0 + 1)
            sig_prop['f0_over_time'] = f0
            sig_prop['f_max_bin'] = int(np.ceil(((sig_prop['f0_range'][-1] * (num_harmonics + 0.5)) / (fs / 2)) * (dft_props['nfft'] / 2)))
            sig_prop['f_max_hz'] = sig_prop['f_max_bin'] * delta_f

        sce_prop = {'names_scf_estimators': names_scf_estimators, 'dft_props': dft_props,
                    'alpha_min_hz': sig_prop['alpha_min_hz'], 'alpha_max_hz': sig_prop['alpha_max_hz'],
                    'delta_alpha_dict': sig_prop['delta_alpha_dict'], 'normalize_scf_to_1': True,
                    'coherence': False}

        return sce_prop, sig_prop

    @staticmethod
    def compute_2d_scf_each_realization_inner(y_sig_samples_list, dft_props, sce_prop):
        # Evaluate 2d spectral correlation function both from each realization of the signal (in y_sig_samples_list)
        # SCFs = spectral correlation functions

        SCE = sce.SpectralCorrelationEstimator(dft_props=dft_props)
        estimated_scfs_and_freqs_list = []
        for yy in y_sig_samples_list:
            temp = SCE.run_spectral_correlation_estimators(x=yy, **sce_prop)
            estimated_scfs_and_freqs_list.append(temp)

        return estimated_scfs_and_freqs_list

    @classmethod
    def compute_2d_scf(cls, sample_paths_list, dft_props, scf_cfg, f0):
        sce_prop_amp, sig_props = cls.prepare_parameters_2d_scf(sample_paths_list, **scf_cfg, f0=f0)
        estimated_scfs_and_freqs_list_amp = cls.compute_2d_scf_each_realization_inner(sample_paths_list, dft_props,
                                                                                      sce_prop_amp)
        scf_dict_first_path = estimated_scfs_and_freqs_list_amp[0]
        scf_estimator_name = sce_prop_amp['names_scf_estimators'][0]

        scf_dict_avg = copy.deepcopy(scf_dict_first_path)
        scf_dict_avg[scf_estimator_name]['scf'] = np.mean(
            np.array([scf_dict[scf_estimator_name]['scf'] for scf_dict in estimated_scfs_and_freqs_list_amp]), axis=0)

        return scf_dict_first_path, scf_dict_avg, sig_props

    @staticmethod
    def plot_2d_scf(scf_dict, sig_prop, amp_range, figsize=None, title=None, show_figures=True, return_tick_labels=False):

        pl = plotter.Plotter(which_plots={'2d': True}, amp_range=amp_range, show_figures=show_figures)
        pl.update_data(sig_props=sig_prop)
        f2ds, xlabels, ylabels = pl.plot_2d_scf(scf_dict, figsize=figsize, title=title)

        if return_tick_labels:
            return f2ds, xlabels, ylabels
        else:
            return f2ds