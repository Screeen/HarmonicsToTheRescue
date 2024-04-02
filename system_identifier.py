import copy
import numpy as np
import scipy


class SystemIdentifier:
    def __init__(self, sig_props, spectral_bins_harmonics, cyclic_bins_harmonics_dict, H_true=None):

        f0_range = sig_props['f0_range']
        self.f0_range = f0_range
        # self.delta_alpha = sig_props['delta_alpha']
        # self.delta_f = sig_props['delta_f']
        self.H_true = H_true  # debug only
        self.max_harmonics = sig_props['num_harmonics']

        self.cyclic_bins_harmonics_dict = copy.deepcopy(cyclic_bins_harmonics_dict)
        self.spectral_bins_harmonics = spectral_bins_harmonics

    def run(self, S_in_in_dict, S_out_in_dict, nfft, names_h_estimators, names_scf_estimators, x=None, y=None,
            S_out_out_dict=None):
        """
        Run system identification algorithms.
        Shape of H_hat: (nfft // 2 + 1, num_h_estimators, num_scf_estimators)
        """
        num_h_estimators = len(names_h_estimators)
        num_scf_estimators = len(self.cyclic_bins_harmonics_dict)
        assert len(S_in_in_dict) == len(S_out_in_dict) == num_scf_estimators
        H_hat = np.zeros((nfft // 2 + 1, num_h_estimators, num_scf_estimators), dtype=complex)

        if not any(names_scf_estimators):  # if spectral correlation estimate not available, sys id is not possible
            return H_hat

        # Frequency-domain wiener estimate
        names_scf_estimators = list(self.cyclic_bins_harmonics_dict.keys())
        first_scf_estimator = names_scf_estimators[0]
        H_hat_wiener = S_out_in_dict[first_scf_estimator]['psd'] / S_in_in_dict[first_scf_estimator]['psd']
        # H_hat_wiener = S_out_out_dict[first_scf_estimator]['psd'] / S_out_in_dict[first_scf_estimator]['psd'].conj()

        for ee, name_h_estimator in enumerate(names_h_estimators):
            H_hat[:, ee] = copy.deepcopy(H_hat_wiener[..., np.newaxis])

        # Time-domain wiener estimate
        try:
            time_wiener_idx = names_h_estimators.index('Time-domain Wiener')
            h_time = self.system_identification_wiener_time_domain(s=x, y=y, num_samples_h=nfft, N_num_samples=len(x))
            H_hat[:, time_wiener_idx] = np.fft.rfft(h_time, n=nfft)
        except ValueError:
            pass
            # print(f"Time-domain Wiener not found among \"name_h_estimators\". Skipping...")

        for name_scf_estimator in names_scf_estimators:

            cyclic_bins_harmonics = copy.deepcopy(self.cyclic_bins_harmonics_dict[name_scf_estimator])
            S_in_in = S_in_in_dict[name_scf_estimator]
            S_out_in = S_out_in_dict[name_scf_estimator]
            if S_out_out_dict is not None:
                S_out_out = S_out_out_dict[name_scf_estimator]

            coherence_or_scf = 'coherence' if 'coherence' in S_out_in else 'scf'

            scf_estimator_idx = names_scf_estimators.index(name_scf_estimator)

            num_cyclic_freqs = len(S_out_in_dict[name_scf_estimator]['alphas'])
            cyclic_bins_harmonics = cyclic_bins_harmonics[cyclic_bins_harmonics < num_cyclic_freqs]

            # Antoni estimate
            try:
                cyclic_wiener_idx = names_h_estimators.index('Antoni')
                for kk in self.spectral_bins_harmonics:
                    if kk == 0:
                        continue  # skip DC (cyclic estimator does not make sense)

                    # print("DEBUG")
                    if name_scf_estimator != 'sample_cov':  # remove 0 from cyclic_bins_harmonics
                        cyclic_bins_harmonics = cyclic_bins_harmonics[cyclic_bins_harmonics > 2]
                    # else:
                    #     cyclic_bins_harmonics = cyclic_bins_harmonics[cyclic_bins_harmonics != kk]

                    # which_harmonic = int(fs * kk / nfft / self.f0_range[1])  # just for debug
                    # cyc_freq_power_kk = (np.abs(S_in_in['coherence'][kk, cyclic_bins_harmonics]))  # weights
                    cyc_freq_power_kk = S_out_in[coherence_or_scf][kk, cyclic_bins_harmonics] ** 2  # weights
                    # cyc_freq_power_kk = (np.abs(S_out_out['coherence'][kk, cyclic_bins_harmonics])) ** 2 # weights
                    cyc_freq_power_kk = cyc_freq_power_kk / np.sum(cyc_freq_power_kk)  # normalize weights

                    estimates_all = S_out_in['scf'][kk, cyclic_bins_harmonics] / S_in_in['scf'][kk, cyclic_bins_harmonics]

                    # Max abs value: 5. Give 0 weight to estimates with large values.
                    indices_large_estimate = np.where(np.abs(estimates_all) > 5)
                    cyc_freq_power_kk[indices_large_estimate] = 0

                    H_hat[kk, cyclic_wiener_idx, scf_estimator_idx] = np.average(estimates_all, weights=cyc_freq_power_kk)

            except ValueError:
                pass

            # Gardner estimate (use only fundamental frequency)
            try:
                gardner_idx = names_h_estimators.index('Gardner')

                if name_scf_estimator != 'sample_cov':  # keep only fundamental frequency
                    cyclic_bins_harmonics = cyclic_bins_harmonics[cyclic_bins_harmonics > 2]

                    # Keep initial consecutive values only
                    idx = np.where(np.diff(cyclic_bins_harmonics) != 1)[0]
                    if len(idx) > 1:
                        cyclic_bins_harmonics = cyclic_bins_harmonics[:idx[1] + 1]

                for kk in self.spectral_bins_harmonics:
                    if kk == 0:
                        continue  # skip DC (cyclic estimator does not make sense)

                    # which_harmonic = int(fs * kk / nfft / self.f0_range[1])  # just for debug
                    # cyc_freq_power_kk = (np.abs(S_in_in['coherence'][kk, cyclic_bins_harmonics]))  # weights
                    cyc_freq_power_kk = S_out_in[coherence_or_scf][kk, cyclic_bins_harmonics] ** 2  # weights
                    # cyc_freq_power_kk = (np.abs(S_out_out['coherence'][kk, cyclic_bins_harmonics])) ** 2 # weights
                    cyc_freq_power_kk = cyc_freq_power_kk / np.sum(cyc_freq_power_kk)  # normalize weights

                    estimates_all = S_out_in['scf'][kk, cyclic_bins_harmonics] / S_in_in['scf'][
                        kk, cyclic_bins_harmonics]

                    # Max abs value: 5. Give 0 weight to estimates with large values.
                    indices_large_estimate = np.where(np.abs(estimates_all) > 5)
                    cyc_freq_power_kk[indices_large_estimate] = 0

                    H_hat[kk, gardner_idx, scf_estimator_idx] = np.average(estimates_all, weights=cyc_freq_power_kk)

            except ValueError:
                pass

        return H_hat

    @staticmethod
    def system_identification_wiener_time_domain(s, y, num_samples_h, N_num_samples, eps=1e-10):

        # https://dspcookbook.github.io/optimal-filtering/wiener-filter-2/
        # https://www.mdpi.com/2076-3417/11/17/7774
        # Last column of s_mat is most recent sample
        num_taps = 2 * num_samples_h + 1
        s_mat = np.zeros((N_num_samples - num_taps, num_taps))
        for jj in range(num_taps):
            s_mat[:, -jj] = s[jj:N_num_samples - num_taps + jj]
        Rss = (1. / (N_num_samples - num_taps)) * s_mat.T @ s_mat
        y_cut = y[num_taps:, None]
        Rsy = np.mean(s_mat * y_cut, axis=0)

        h_time = scipy.linalg.solve(Rss, Rsy, assume_a='her')[1:]
        h_time = h_time / np.max(np.abs(h_time) + eps)

        return h_time
