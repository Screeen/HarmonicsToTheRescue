import numba
import numpy as np
import scipy
import utils as u
import manager as si_manager


m = si_manager.Manager()


class SpectralCorrelationEstimator:

    def __init__(self, dft_props):

        win_name = 'hann'
        # win_name = 'cosine'
        win = scipy.signal.windows.get_window(win_name, dft_props['nw'], fftbins=True)
        # print(f"{win_name = } selected for spectral correlation estimation")

        if win_name == 'cosine' and dft_props['noverlap'] != dft_props['nw'] // 2:
            raise ValueError(f'For {win_name = }, {dft_props["noverlap"] = } must be {dft_props["nw"] // 2 = }')

        self.acp_window = win / np.sqrt(np.sum(win ** 2))  # normalize to unit power

        # Condition 3.3.1 in "Cyclic spectral analysis in practice (2007)": power calibration
        assert np.isclose(np.sum(self.acp_window ** 2), 1, atol=1e-3)

        self.dft_props = dft_props

    def run_spectral_correlation_estimators(self, names_scf_estimators, x, dft_props, alpha_max_hz, delta_alpha_dict,
                                            alpha_min_hz=0, y=None, normalize_scf_to_1=False, coherence=False):

        fs = dft_props['fs']
        nfft = dft_props['nfft']
        nw = dft_props['nw']
        noverlap_samples = dft_props['noverlap']

        # Make a dict containing a dict for each estimator. The key is the name of the estimator.
        res = {estimator_name: dict() for estimator_name in names_scf_estimators}

        if y is None:
            y = x
        else:
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]

        sample_cov_dict = {'window': self.acp_window, 'fs_': fs, 'Nw_': nw, 'noverlap_samples_': noverlap_samples,
                           'complex_stft': False, 'phase_correction': True}
        x_stft_gio = m.get_stft_phase_corrected(x, **sample_cov_dict)
        if y is not None:
            y_stft_gio = m.get_stft_phase_corrected(y, **sample_cov_dict)
        else:
            y_stft_gio = x_stft_gio

        # Rescaling so that np.diag(sample_cov) == psd using scipy
        x_stft_gio[1:-1] *= np.sqrt(2)
        y_stft_gio[1:-1] *= np.sqrt(2)

        s_sample_cov = x_stft_gio @ y_stft_gio.conj().T / x_stft_gio.shape[1]
        cpsd_sample_cov = np.diag(s_sample_cov)

        if 'sample_cov' in names_scf_estimators:
            freqs_sample_cov = np.fft.rfftfreq(nfft, 1 / fs)
            res['sample_cov']['freqs'] = freqs_sample_cov
            res['sample_cov']['alphas'] = freqs_sample_cov
            res['sample_cov']['scf'] = s_sample_cov

        if 'dirichlet' in names_scf_estimators:
            freqs_dirichlet, alphas_dirichlet, s_scf_dirichlet = (
                self.compute_dirichlet_cyclic_periodogram(x, y=y, fs_=fs, Nw_=nw, alpha_max=alpha_max_hz,
                                                          win=self.acp_window))

            res['dirichlet']['freqs'] = freqs_dirichlet
            res['dirichlet']['alphas'] = alphas_dirichlet
            res['dirichlet']['scf'] = s_scf_dirichlet

        if 'acp' in names_scf_estimators:
            freqs_acp, alphas_acp, s_acp, coherence_acp = (
                self.compute_averaged_cyclic_periodogram(x, y=y, fs_=fs, Nw=nw, nfft=nfft,
                                                         noverlap_samples_=noverlap_samples,
                                                         alpha_min_hz=alpha_min_hz,
                                                         alpha_max_hz=alpha_max_hz,
                                                         conjugate_scf=False,
                                                         window=self.acp_window,
                                                         delta_alpha=delta_alpha_dict['acp'],
                                                         compute_coherence=coherence))
            res['acp']['freqs'] = freqs_acp
            res['acp']['alphas'] = alphas_acp
            res['acp']['scf'] = s_acp
            res['acp']['coherence'] = coherence_acp

        if 'psd' in names_scf_estimators:
            psd = scipy.signal.csd(y, x, fs=fs,
                                   nperseg=nw,
                                   nfft=nfft,
                                   noverlap=noverlap_samples,
                                   return_onesided=True, scaling='spectrum', detrend=False)[1]

            res['psd']['psd'] = psd
            res['psd']['freqs'] = np.empty((0,))
            res['psd']['alphas'] = np.empty((0,))
            res['psd']['scf'] = np.empty((0,))

        if normalize_scf_to_1:
            for estimator_name in res.keys():
                if res[estimator_name]['scf'] is not None:
                    res[estimator_name]['scf'] = res[estimator_name]['scf'] / np.max(res[estimator_name]['scf'])

        # Use the same PSD for all estimators, so that we can compare them more easily.
        for key in res.keys():
            res[key]['psd'] = cpsd_sample_cov
            # if key != 'sample_cov' and key != 'psd':
            #     res[key]['scf'][:, 0] = cpsd_sample_cov

        return res

    @staticmethod
    def compute_averaged_cyclic_periodogram(x, y=None, fs_=16000, Nw=512, noverlap_samples_=0, window=None,
                                            alpha_min_hz=0.0, alpha_max_hz=500., delta_alpha=None,
                                            conjugate_scf=False, nfft=None, compute_coherence=False):
        """
        Compute time-smoothed averaged cyclic periodogram (ACP) between x and y.
        If y is not provided, then the ACP is computed between x and x.

        Cyclic cross-spectrum of X and Y:
        S_(xy)(f, alpha) = E[X(f) Y(f - alpha).conj()]
        where X(f) is the STFT of x(t) and Y(f - alpha) is the STFT of y(t) modulated by exp(j 2 pi alpha t)

        Squared cyclic coherence of X and Y:
        gamma^2_(xy)(f, alpha) = |S_(xy)(f, alpha)|^2 / (S_x(f) S_y(f - alpha)),
        where S_x(f) is the PSD of x(t) and S_y(f - alpha) is the PSD of y(t) modulated by exp(j 2 pi alpha t)
        """

        if nfft is None:
            nfft = Nw

        def local_stft(y_):
            _, _, Y_ = scipy.signal.stft(y_, fs=fs_, window=window, nperseg=Nw, noverlap=noverlap_samples_,
                                         detrend=False, return_onesided=False, boundary=None, padded=False, axis=-1)
            return Y_

        N_num_samples = len(x)
        nfft_real = nfft // 2 + 1
        # R_shift_samples = Nw - noverlap_samples_
        # L_num_frames = np.floor((N_num_samples - Nw + R_shift_samples) / R_shift_samples).astype(int)
        squared_coherence = None

        delta_alpha_min = fs_ / N_num_samples  # minimum possible delta_alpha
        if delta_alpha is not None:  # provided by user
            if delta_alpha < delta_alpha_min:
                raise ValueError(f'delta_alpha={delta_alpha} is too small. Cannot be smaller than {delta_alpha_min}')
        else:
            delta_alpha = delta_alpha_min

        # Compute vector of cyclic frequencies of interest (in Hz)
        alpha_vec_hz = np.arange(alpha_min_hz, alpha_max_hz, delta_alpha)
        if len(alpha_vec_hz) < 2:
            raise ValueError(f'{alpha_max_hz=} is too small. {delta_alpha=}')

        # Compute vector of frequencies of interest (in Hz)
        f_vec_hz = np.fft.fftfreq(nfft, d=1 / fs_)
        f_vec_hz_real = np.abs(f_vec_hz[:nfft_real])

        y_alpha_time = SpectralCorrelationEstimator.modulate_signal_all_alphas_vec(N_num_samples, alpha_vec_hz, fs_, y)
        y_alpha_stft = local_stft(y_alpha_time)

        # Corresponds to X(f-alpha) for alpha=0 -> X(f)
        # Rescale so that np.diag(sample_cov) == psd using scipy
        x_stft = local_stft(x)
        x_stft[1:-1] *= np.sqrt(2)
        y_alpha_stft[:, 1:-1] *= np.sqrt(2)

        # Discard values that correspond to negative spectral frequencies, ff_hz < 0
        x_stft = x_stft[:nfft_real]
        y_alpha_stft = y_alpha_stft[:, :nfft_real]

        # Compute time-smoothed averaged periodogram (cyclic correlation)
        if not conjugate_scf:
            cyclic_correlation = np.mean(x_stft[np.newaxis] * y_alpha_stft.conj(), axis=-1).T
        else:
            cyclic_correlation = np.mean(x_stft[np.newaxis] * y_alpha_stft, axis=-1).T

        if compute_coherence:
            if conjugate_scf: raise NotImplementedError("coherence and conjugate_scf are not implemented together")

            x_psd = np.mean(np.square(np.abs(x_stft)), axis=-1)
            y_alpha_psd = np.mean(np.square(np.abs(y_alpha_stft)), axis=-1)
            normalizations = x_psd[np.newaxis] * y_alpha_psd
            squared_coherence = np.abs(cyclic_correlation) ** 2 / normalizations.T

        # Set to 0 values that correspond to negative frequencies in SCF: (ff_hz - aa_hz) < 0
        # Leaving this untouched is like having Hermitian symmetry in the sample estimate of the SCF,
        # and it can be convenient for system identification.
        # for idx_ff, ff_hz in enumerate(f_vec_hz_real):
        #     for idx_aa, aa_hz in enumerate(alpha_vec_hz):
        #         if ff_hz - aa_hz < 0:
        #             cyclic_correlation[idx_ff, idx_aa] = 0

        return f_vec_hz_real, alpha_vec_hz, cyclic_correlation, squared_coherence

    @staticmethod
    @numba.jit(cache=True, nopython=True)
    def modulate_signal_all_alphas(L_, alpha_vec_hz, fs_, y):
        # Compute x_stft for different frequency shifts X(f - alpha),
        # meaning that X is shifted to HIGHER frequencies.
        # Modulation property: X(f - f0) <=> x(t) * exp(j 2 pi f0 t)
        y_alpha_time = np.zeros((len(alpha_vec_hz), L_), dtype=np.complex128)
        time_axis = 2j * np.pi * np.arange(L_) / fs_
        for idx_aa, aa_hz in enumerate(alpha_vec_hz):
            # Equivalent to y_hat[n] = y[n] * np.exp(j 2 pi alpha_hz n / fs)
            y_alpha_time[idx_aa, :] = y * np.exp(aa_hz * time_axis)
        return y_alpha_time

    @staticmethod
    @numba.jit(cache=True, nopython=True, parallel=True)
    def modulate_signal_all_alphas_vec(L_: int, alpha_vec_hz: np.ndarray, fs_: float, y: np.ndarray) -> np.ndarray:
        time_axis = 2j * np.pi * np.arange(L_) / fs_
        alpha_matrix = np.exp(np.outer(alpha_vec_hz, time_axis))
        y_alpha_time = y * alpha_matrix
        return y_alpha_time

    @staticmethod
    def compute_dirichlet_cyclic_periodogram(x, y=None, fs_=16000, Nw_=512, window_name='hann',
                                             alpha_min=0.0, alpha_max=0.5, win=None):
        """ A faster algorithm for the calculation of the fast spectral correlation"""

        complex_stft = False

        def local_stft(y_, win_):
            return u.stft(y_, window=win_, fs_=fs_, Nw_=Nw_, noverlap_samples_=noverlap_samples_,
                          complex_stft=complex_stft, padding=False)

        if win is None:
            raise ValueError("Provide window (e.g. scipy.signal.windows.hann(Nw_, sym=False)")

        R_shift_samples_ = int(fs_ / (2 * alpha_max))  # R
        P_num_scanning_correlations = int((Nw_ - 1) / (2 * R_shift_samples_))  # P
        L_num_frames_ = int(1 + np.floor((len(x) - Nw_) / R_shift_samples_))  # M, this should correspond to X.shape[1]
        noverlap_samples_ = Nw_ - R_shift_samples_

        # Compute the Dirichlet kernel
        # TODO: Are -P and +P actually included? (I think so)
        # p_vector = np.arange(-P_num_scanning_correlations, P_num_scanning_correlations + 1)[..., np.newaxis]
        n_vector = np.arange(0, Nw_)[np.newaxis, ...]
        dirichlet_arg = 2 * np.pi * (n_vector - Nw_ / 2) / Nw_
        # dirichlet_kernel = np.sum(np.exp(1j * p_vector * dirichlet_arg), axis=0)
        # dirichlet_kernel_old = dirichlet_kernel.real

        dirichlet_kernel = scipy.special.diric(dirichlet_arg, 2 * P_num_scanning_correlations + 1)[0]

        # Compute the standard STFT
        # TODO Is this a real or a complex STFT? (I think complex)
        X = local_stft(x, win)

        # Compute the modified STFT
        win_mod = win * dirichlet_kernel
        Yd = local_stft(y, win_mod)

        # Computation of the normalisation parameter
        dft_size_normalization = L_num_frames_ * R_shift_samples_
        # TODO Is this a real or a complex FFT? (I think complex)
        Normalization_dft = np.fft.fft(win ** 2 * dirichlet_kernel.real, n=dft_size_normalization)

        # TODO Should I discard the first bin? In the block-diagram in the paper it says a = 1, ..., M
        Normalization_dft = L_num_frames_ * Normalization_dft[:L_num_frames_]

        # In the paper, they have (time-frames, frequency bins) as the shape of X, but we have (frequency, time-frames).
        # DFT needs to transform time-frames --> cyclic frequencies.
        fsc = np.fft.fft(X * np.conj(Yd), axis=-1, n=L_num_frames_)
        fsc = fsc[:, :L_num_frames_] / Normalization_dft[None, :]

        f_vec_hz_real = np.fft.rfftfreq(Nw_, d=1 / fs_)
        alpha_vec_hz = SpectralCorrelationEstimator.get_alpha_vec_hz_dirichlet(L_num_frames_, R_shift_samples_, fs_,
                                                                               alpha_max)
        fsc = fsc[:, alpha_vec_hz >= 0]
        alpha_vec_hz = alpha_vec_hz[alpha_vec_hz >= 0]
        alpha_vec_hz = alpha_vec_hz[alpha_vec_hz < alpha_max]
        fsc = fsc[:len(f_vec_hz_real), :len(alpha_vec_hz)]

        # Set to 0 values that correspond to negative frequency differences in SCF: (ff_hz - aa_hz) < 0
        # for idx_ff, ff_hz in enumerate(f_vec_hz_real):
        #     for idx_aa, aa_hz in enumerate(alpha_vec_hz):
        #         if ff_hz - aa_hz < 0:
        #             fsc[idx_ff, idx_aa] = 0

        return f_vec_hz_real, alpha_vec_hz, fsc

    @staticmethod
    def compute_averaged_cyclic_periodogram_symmetric(x, y=None, fs_=16000, nfft_=512, noverlap_samples_=0,
                                                      window_name='hann',
                                                      alpha_min=0.0, alpha_max=0.5):

        raise NotImplementedError("Use the non-symmetric version, this one is not well tested.")

    """
        stft_padding = True
        f_frequencies, t_times, _ = scipy.signal.stft(x, fs=fs_, window=window_name, nperseg=nfft_,
                                                      noverlap=noverlap_samples_, return_onesided=False)

        N_num_samples = len(x)
        L_num_frames_ = len(t_times)
        K_num_freqs = len(f_frequencies)
        delta_alpha = fs_ / N_num_samples

        A_num_cyc_freqs = int(alpha_max / delta_alpha)
        if A_num_cyc_freqs % 2 == 0:  # make them odd so that 0 is included
            A_num_cyc_freqs += 1
        alpha_cyclic_freqs = np.linspace(-alpha_max / 2, alpha_max / 2, A_num_cyc_freqs, endpoint=True)
        shift_samples = nfft_ - noverlap_samples_

        ic(L_num_frames_, K_num_freqs, N_num_samples, delta_alpha, shift_samples)

        x_stft_freq_shift = np.zeros((A_num_cyc_freqs,) + (K_num_freqs, L_num_frames_), dtype=np.complex128)
        for idx_alpha, alpha_hz in enumerate(alpha_cyclic_freqs):
            x_mod = x * np.exp(2j * np.pi * alpha_hz * np.arange(N_num_samples) / fs_)
            x_stft_freq_shift[idx_alpha, :] = u.stft(x_mod, Nw_=nfft_, noverlap_samples_=noverlap_samples_,
                                                     complex_stft=True, padding=stft_padding)

        # Set to 0 all values of x_stft_min_alpha that correspond to (ff_hz - aa_hz) < 0
        for idx_ff, ff_hz in enumerate(f_frequencies):
            for idx_aa, aa_hz in enumerate(alpha_cyclic_freqs):
                if ff_hz - aa_hz < 0 or ff_hz + aa_hz > fs_ / 2:
                    x_stft_freq_shift[idx_aa, idx_ff, :] = 0

        # The cyclic periodogram is defined as
        # I(alpha, f, t) = (1 / L_num_frames_) X(f + alpha/2, t) X.conj()(f - alpha/2, t)
        # where X(f, t) is the STFT of x(t).
        cp = np.zeros((A_num_cyc_freqs,) + (K_num_freqs, L_num_frames_), dtype=np.complex128)
        for idx_alpha, alpha_hz in enumerate(alpha_cyclic_freqs):
            idx_alpha_rev = A_num_cyc_freqs - idx_alpha - 1
            cp[idx_alpha, :] = x_stft_freq_shift[idx_alpha_rev, :] * x_stft_freq_shift[idx_alpha,
                                                                     :].conj() / N_num_samples

        # Phase correction factor (depends on alpha and time-frame index)
        delays_samples = np.arange(L_num_frames_) * shift_samples
        alpha_distances_hz = 2 * np.abs(alpha_cyclic_freqs)
        phase_corr = np.exp(-2j * np.pi * alpha_distances_hz[:, None, None] * delays_samples[None, None, :] / fs_)
        phase_corr = np.ones_like(phase_corr)

        # apply phase correction and Sum over time-frames. New shape is (len(alpha_cyclic_freqs), K_num_freqs)
        scf = np.sum(cp * phase_corr, axis=-1) / L_num_frames_

        # Discard part corresponding to negative frequencies and negative cyclic frequencies
        scf = scf[A_num_cyc_freqs // 2:, :K_num_freqs // 2 - 1]
        scf = scf.T

        alpha_cyclic_freqs_asymmetric = 2 * alpha_cyclic_freqs[A_num_cyc_freqs // 2:]

        return f_frequencies[:K_num_freqs // 2 - 1], alpha_cyclic_freqs_asymmetric, scf
    """

    @staticmethod
    def get_alpha_vec_hz_dirichlet(L_num_frames_, R_shift_samples_, fs_, alpha_max=None):

        # delta_alpha = fs_ / (L_num_frames_ * R_shift_samples_)
        alpha_vec_hz = np.fft.fftfreq(L_num_frames_, d=R_shift_samples_ / fs_)

        if len(alpha_vec_hz) == 1 or alpha_vec_hz[1] <= 0:
            return np.empty((0,))

        # so that delta_alpha = alpha_vec_hz[1] - alpha_vec_hz[0] = fs_ / (L_num_frames_ * R_shift_samples_)
        return alpha_vec_hz

