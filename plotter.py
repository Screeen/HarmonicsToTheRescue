import numpy as np
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter, NullFormatter

import utils as u
import warnings
import librosa
from pathlib import Path
import os
import datetime

u.set_printoptions_numpy()
u.set_plot_options()
current_folder = Path(os.getcwd())
date_y_m_d = datetime.datetime.now().strftime("%Y-%m-%d")
folder_str = u.check_create_folder(folder_name=date_y_m_d, parent_folder=current_folder / 'figs')
folder = Path(folder_str)


class Plotter:
    num_reps = 3
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'] * num_reps
    linestyles = ['-', '--', ':', '-.', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':'] * num_reps
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
              'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] * num_reps

    def __init__(self, which_plots, save_figures=False, log_transform_2d_plots=True,
                 show_figures=True, amp_range=(None, None), names_scf_estimators=None, names_h_estimators=None,
                    indices_h_estimators=None):

        self.mse_db_max_error = 50
        self.mse_db_min_error = -50

        self.which_plots = which_plots
        self.save_figures = save_figures
        self.show_figures = show_figures
        self.log_transform_2d_plots = log_transform_2d_plots
        self.amp_range = amp_range

        self.dft_props = {}
        self.sig_props = {}
        self.sys_props = {}

        self.ref_sig = None
        self.ref_sig_psd = None
        self.evaluated_bins = None

        self.names_scf_estimators = names_scf_estimators
        self.names_h_estimators = names_h_estimators
        self.indices_h_estimators = indices_h_estimators


    def update_data(self, dft_props=None, sig_props=None, sys_props=None,
                    ref_sig=np.empty(0), ref_sig_psd=np.empty(0),
                    evaluated_bins=np.empty(0)):

        if dft_props:
            self.dft_props.update(dft_props)

        if sig_props:
            self.sig_props.update(sig_props)

        if sys_props:
            self.sys_props.update(sys_props)

        if ref_sig.any():
            self.ref_sig = ref_sig

        if ref_sig_psd.any():
            self.ref_sig_psd = ref_sig_psd

        if evaluated_bins.any():
            self.evaluated_bins = evaluated_bins


    @staticmethod
    def determine_f0_range(Nw_nfft, f0):

        f0_min, f0_max = (0, 300)  # limits of human voice
        if Nw_nfft >= 128 and (f0 > f0_max or f0 < f0_min):
            raise ValueError(f'{f0=} Hz is outside the range [{f0_min}, {f0_max}] Hz')
        elif Nw_nfft < 128 and (f0 > f0_max or f0 < f0_min):
            # debugging
            f0_max = 2 * f0
            f0_min = 0

        return f0_min, f0_max

    def find_peaks_psd(self, y, sig_props, dft_props, peak_finding_loud_bins=True):

        if not self.which_plots['1d']:
            return None, None, sig_props

        # Find peaks in PSD, corresponding to frequency bins which we want to plot as function of alpha,
        # and f0 estimate.
        # if sig_props['simulated_signal']:
        #     f0_min_max = sig_props['f0_range'][0], sig_props['f0_range'][-1]
        # else:
        #     f0_min_max = self.determine_f0_range(Nw_nfft=dft_props['nfft'], f0=sig_props['f0'])
        f0_min_max = sig_props['f0_range'][0], sig_props['f0_range'][-1]

        fs = dft_props['fs']
        Nw_nfft = dft_props['nfft']
        f0 = sig_props['f0_range'][1]  # mean value
        num_harmonics = sig_props['num_harmonics']
        f0_min, f0_max = f0_min_max
        f_max_bin = sig_props['f_max_bin']
        simulated_signal = sig_props['simulated_signal']

        def find_peaks_internal(prominence_, distance_):
            peaks_loc__, peaks_properties__ = scipy.signal.find_peaks(
                psd, prominence=prominence_, distance=distance_)
            locs__ = peaks_loc__ < f_max_bin
            peaks_loc__ = peaks_loc__[locs__]
            peaks_properties__['prominences'] = peaks_properties__['prominences'][locs__]
            return peaks_loc__, peaks_properties__

        psd = scipy.signal.welch(y, fs=fs, nperseg=Nw_nfft, noverlap=dft_props['noverlap'], return_onesided=True)[1]
        if peak_finding_loud_bins:
            print(f"Use peak finding for f0={f0:.0f} Hz, {num_harmonics} harmonics, {f0_min:.0f} <= f0 <= {f0_max:.0f} Hz")

            prominence = np.max(psd) / 200
            distance = np.ceil(0.2 * fs / Nw_nfft)
            peaks_loc_, peaks_properties = find_peaks_internal(prominence, distance)

            max_iterations_peak_finding = 10
            for ii in range(2, max_iterations_peak_finding):
                if len(peaks_loc_) > 8 or (simulated_signal and len(peaks_loc_) >= num_harmonics):
                    break

                peaks_loc_, peaks_properties = find_peaks_internal(prominence / ii, np.maximum(1, int(distance / ii)))

            # Sort peaks according to peaks_properties['prominences']
            # peaks_loc_ = peaks_loc_[np.argsort(peaks_properties['prominences'])[::-1]]
            peaks_loc_ = peaks_loc_[:10] if not simulated_signal else peaks_loc_[:num_harmonics]

            f0_est_ = fs / 2
            f0_est_bin = Nw_nfft // 2
            for ii, peak_loc_ in enumerate(peaks_loc_):
                peak_hz_ = peak_loc_ / Nw_nfft * fs
                if peak_hz_ < f0_est_ and f0_min <= peak_hz_ <= f0_max:
                    f0_est_ = peak_hz_
                    f0_est_bin = peak_loc_

            peaks_loc_ = peaks_loc_[peaks_loc_ >= f0_est_bin]

        else:
            print(f"Retrieve frequencies of harmonics and convert them to bins")
            f0_est_ = f0
            peaks_loc_ = [np.round(x).astype(int) for x in
                          np.arange(1, num_harmonics + 1) * f0_est_ / (fs / Nw_nfft)]

        peaks_loc_ = sorted(peaks_loc_)[::-1]

        if len(peaks_loc_) == 0:
            warnings.warn(f'No peaks found in PSD for f0={f0:.0f} Hz')

        return peaks_loc_, f0_est_, sig_props

    def plot_all(self, spectral_correlation_functions, f0=None):

        sig_props = self.sig_props
        snr_db = sig_props['snr']
        f_max_bin = sig_props['f_max_bin']
        f_max_hz = sig_props['f_max_hz']
        alpha_max_hz = sig_props['alpha_max_hz']
        simulated_signal = sig_props['simulated_signal']
        f0 = sig_props['f0'] if f0 is None else f0

        ref_sig = self.ref_sig
        evaluated_bins = self.evaluated_bins

        dft_props = self.dft_props
        fs = dft_props['fs']
        nfft = dft_props['nfft']
        R_shift_samples = dft_props['nfft'] - dft_props['noverlap']

        f0_title = f'{f0:.0f}'
        synth_or_real = 'synthetic' if simulated_signal else 'real'

        if self.which_plots['f0_spectrogram']:
            self.plot_spectrogram_and_f0(ref_sig, fs, R_shift_samples, nfft, sig_props['f0_over_time'])
            u.play(ref_sig, fs=fs, volume=0.3)

        if self.which_plots['time_psd']:
            self.plot_time_and_psd(y=ref_sig, dft_props=dft_props, sig_props=sig_props)

        if self.which_plots['1d']:
            peaks_loc, f0_est, sig_props = self.find_peaks_psd(y=ref_sig,
                                                               sig_props=sig_props,
                                                               dft_props=dft_props,
                                                               peak_finding_loud_bins=False)

            title_1d = f'Cyclic frequencies for f0={f0_title} Hz, {synth_or_real} signal'

            if peaks_loc is None:
                raise ValueError('peaks_loc must be provided if 1d plots are to be shown')
            alpha_max_hz_quantized = alpha_max_hz + fs / nfft
            peaks_loc_hz = [peak_loc / nfft * fs for peak_loc in peaks_loc]
            for scf_name in spectral_correlation_functions.keys():
                title = title_1d + f", {scf_name}"
                f1d = Plotter.plot_matrix_rows(peaks_loc=peaks_loc, peaks_loc_hz=peaks_loc_hz,
                                               title=title,
                                               alpha_max_hz_quantized=alpha_max_hz_quantized,
                                               s_scf_name=scf_name,
                                               est_scf=spectral_correlation_functions[scf_name])
                if self.save_figures:
                    u.savefig(f1d, folder / f'scf_1d_{f0:.0f}hz_{scf_name}_{synth_or_real}.pdf')
                plt.close(f1d)
                plt.pause(0.01)

        if self.which_plots['2d']:
            self.plot_2d_scf(spectral_correlation_functions, f0)

        if self.which_plots['3d']:
            title_2d_3d = f'Estimated SCF for f0={f0_title} Hz, {synth_or_real} signal'
            for scf_name in spectral_correlation_functions.keys():

                freqs = spectral_correlation_functions[scf_name]['freqs'][:f_max_bin]
                if scf_name == 'sample_cov':
                    scf = spectral_correlation_functions[scf_name]['scf'][:f_max_bin, :f_max_bin]
                    alphas = freqs
                    alpha_max_hz = f_max_hz
                    xy_label_3d = ('Frequency 1 [Hz]', 'Frequency 2 [Hz]')
                else:
                    scf = spectral_correlation_functions[scf_name]['scf'][:f_max_bin, :].T
                    alphas = spectral_correlation_functions[scf_name]['alphas']
                    xy_label_3d = ('Frequency [Hz]', 'Cyclic frequency [Hz]')

                f3d = u.plot_surface(z=scf,
                                     x=freqs,
                                     y=alphas,
                                     xlim=[0, f_max_hz],
                                     ylim=[0, alpha_max_hz],
                                     title=title_2d_3d + f", {scf_name}",
                                     xy_label=xy_label_3d,
                                     show_figures=self.show_figures)

                if self.save_figures:
                    u.savefig(f3d, folder / f"scf_3d_{f0:.0f}hz_{scf_name}_{synth_or_real}.pdf")
                plt.close(f3d)
                plt.pause(0.01)

        if self.which_plots['estimated_h_time']:
            self.plot_h_h_hat_time_domain(self.sys_props['h'], self.sys_props['H_hat'], self.names_h_estimators,
                                          self.indices_h_estimators, nfft)

        if self.which_plots['estimated_h_freq']:
            # Show results for either ACP or Dirichlet (find the corresponding index)
            try:
                scf_estimator_idx = self.names_scf_estimators.index('acp')
            except ValueError:
                scf_estimator_idx = self.names_scf_estimators.index('dirichlet')

            self.plot_h_h_hat_frequency_domain(self.names_h_estimators, self.sys_props['H'][evaluated_bins],
                                               self.sys_props['H_hat'][evaluated_bins, :, scf_estimator_idx],
                                               self.ref_sig_psd[evaluated_bins],
                                               suptitle=f'Estimates for SNR = {snr_db} dB')

    @staticmethod
    def plot_time_psd_internal(y, title_, dft_props, show_figures=True):
        # Plot signal in time domain and its PSD

        fig_, ax_ = plt.subplots(1, 2, figsize=(6., 2.))
        ax_[0].plot(np.arange(len(y)) / dft_props['fs'], y)
        ax_[0].set_xlabel('Time [s]')
        ax_[0].set_ylabel('Amplitude')
        ax_[0].grid(True)
        ax_[1].psd(y, NFFT=dft_props['nfft'],
                   Fs=dft_props['fs'], noverlap=dft_props['noverlap'], scale_by_freq=False)
        ax_[1].set_xlabel('Frequency [Hz]')
        ax_[1].set_ylabel('PSD [dB/Hz]')
        ax_[1].grid(True)

        fig_.suptitle(title_, y=0.9)

        fig_.tight_layout()
        if show_figures:
            fig_.show()
        plt.close(fig_)
        plt.pause(0.01)

        return fig_, ax_

    def plot_time_and_psd(self, y, dft_props, sig_props):

        title_ = 'Waveform and power spectral density (PSD)'
        if sig_props['simulated_signal']:
            title_ += f' - f0={sig_props["f0_range"][1]:.0f} Hz'
        # snr_ = sig_props["snr"]
        # title_ += f' - SNR={snr_ :.0f} dB'

        f, ax = self.plot_time_psd_internal(y, dft_props=dft_props, title_=title_, show_figures=self.show_figures)
        if self.save_figures:
            u.savefig(f, folder / f'time_freq_{sig_props["f0"]:.0f}hz.pdf')

        plt.close(f)
        plt.pause(0.01)

    def plot_error_vs_variation_param(self, names_h_estimators_, rmse, rmse_95, variation_values_x):

        if not 'error_vs_variation_param' in self.which_plots:
            return None

        # Plot rmse as a function of SNR. Also show error bars as faded areas
        fontsize = 12
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3), constrained_layout=True)
        for jj, h_estimator_ in enumerate(names_h_estimators_):
            ax.plot(variation_values_x, rmse[:, jj], marker=self.markers[jj], label=h_estimator_, linestyle=self.linestyles[jj])
            ax.fill_between(variation_values_x, rmse[:, jj] - rmse_95[:, jj], rmse[:, jj] + rmse_95[:, jj], alpha=0.2)
        ax.set_xlabel('SNR [dB]', fontsize=fontsize)
        ax.set_ylabel('RMSE', fontsize=fontsize)
        ax.grid(True)
        ax.legend(fontsize=fontsize)
        ax.set_title('RMSE vs SNR (system identification)', fontsize=fontsize + 1)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        fig.tight_layout()

        if self.show_figures:
            fig.show()
            plt.pause(0.05)
        if self.save_figures:
            u.savefig(fig, folder / 'rmse_vs_snr.pdf')

        return fig

    def plot_h_h_hat_frequency_domain(self, names_h_estimators_, H_nonzero, H_hat_nonzero, S_ss_non_zero, suptitle=''):
        # Plot transfer function in two subplots: one with absolute value of H and H_hat, and one with
        # phase of H and H_hat
        fontsize = 9
        tick_fontsize = 7
        fig, axes = plt.subplots(3, 1, figsize=(3.5, 4.5), constrained_layout=True)

        names_estimators = names_h_estimators_ + ['True H']
        H_all = np.concatenate((H_hat_nonzero, H_nonzero[:, np.newaxis]), axis=1)

        # Store in a dict label, marker, alpha=0.8 and linestyle for each estimator
        estimator_options = dict()
        for h_estimator_idx_, h_estimator in enumerate(names_estimators):
            estimator_options[h_estimator] = dict(label=h_estimator,
                                                  alpha=0.8,
                                                  color=self.colors[h_estimator_idx_],
                                                  linestyle=self.linestyles[0] if h_estimator == 'True H' else self.linestyles[h_estimator_idx_],
                                                  # marker=self.markers[h_estimator_idx_],
                                                  # markeredgecolor=self.colors[h_estimator_idx_],
                                                  # markerfacecolor='none',
                                                  # markeredgewidth=0.5,
                                                  # markersize=3,
            )

        for h_estimator_idx_, estimator_opt in enumerate(estimator_options.values()):
            axes[0].plot(np.real(H_all[:, h_estimator_idx_]), **estimator_opt)
            axes[1].plot(np.imag(H_all[:, h_estimator_idx_]), **estimator_opt)
        axes[0].set_title('Transfer function (real)')
        axes[1].set_title('Transfer function (imag)')

        # On axis 2 plot PSD. Use non_zero_idx as labels. They are in floating point format (frequencies in Hz), so round
        # them to integers to ease the reading.
        # Also they are too many to plot, so plot only a subset of them.
        # TO select only a few, do something like this:

        # if non_zero_frequencies is not None:
        #     non_zero_frequencies = np.round(non_zero_frequencies).astype(int)
        #     non_zero_frequencies = non_zero_frequencies[non_zero_frequencies < 5000]
        #     non_zero_frequencies = non_zero_frequencies[non_zero_frequencies > 0]
        #     non_zero_frequencies = non_zero_frequencies[:20]
        #
        #     axes[2].plot(non_zero_frequencies, np.abs(S_ss_non_zero)[:20], label='S_ss')
        #     axes[2].set_xlabel('Frequency [Hz]', fontsize=fontsize)
        #
        # else:

        axes[2].plot(np.abs(S_ss_non_zero), label='S_ss')
        axes[2].set_title('Input PSD')
        axes[2].set_yticklabels([])


        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            ax.grid(True)
            ax.legend(fontsize=fontsize - 1)
            ax.title.set_size(fontsize)
            # Tick font size
            ax.xaxis.label.set_size(tick_fontsize)
            ax.yaxis.label.set_size(tick_fontsize)

        if suptitle != '':
            fig.suptitle(suptitle, fontsize=fontsize + 1)

        if self.show_figures:
            fig.show()
        if self.save_figures:
            u.savefig(fig, folder / 'h_h_hat_frequency_domain.pdf')

        return fig

    @classmethod
    def plot_matrix_rows(cls, est_scf, peaks_loc, peaks_loc_hz, alpha_max_hz_quantized, s_scf_name, title):
        title_font_size = 14
        font_size = 11  # for tick labels, axis labels, legend

        scf = est_scf['scf']
        x_axis = est_scf['alphas'][est_scf['alphas'] < alpha_max_hz_quantized]
        num_freqs_x_axis = len(x_axis)

        # Plot an interesting bin (high amplitude) using alpha as x-axis.
        fig, ax = plt.subplots(1, 1)
        ax.grid(True)

        # Enable finer grid
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth='0.1', color='black')

        for idx, (peak_loc, peak_hz) in enumerate(zip(peaks_loc, peaks_loc_hz)):

            if peak_hz < alpha_max_hz_quantized:

                if s_scf_name == 'acp' or s_scf_name == 'dirichlet':
                    pp = np.abs(scf[peak_loc, :num_freqs_x_axis].T)
                elif s_scf_name == 'sample_cov':
                    pp = np.roll(scf[peak_loc, :], shift=-peak_loc)
                    pp = pp[:num_freqs_x_axis]
                    pp = np.abs(pp.T)[:num_freqs_x_axis]
                else:
                    raise ValueError(f"Unknown s_scf_name: {s_scf_name}")

                ax.plot(x_axis, pp, markerfacecolor='none',
                        markeredgewidth=0.5,
                        label=f'{peak_hz:.0f} Hz', linestyle=cls.linestyles[idx], marker=cls.markers[idx])

        ax.set_title(title, fontsize=title_font_size)
        x_label = 'Cyclic frequencies [Hz]' if s_scf_name == 'acp' or s_scf_name == 'dirichlet' else 'Frequency 2 [Hz]'
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel('Magnitude (normalized)', fontsize=font_size)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=font_size)

        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.tick_params(axis='both', which='minor', labelsize=font_size)

        fig.show()
        plt.pause(0.01)

        return fig

    @staticmethod
    def plot_h_h_hat_time_domain(h, H_hat, names_h_estimators, indices_h_estimators, nfft):
        # Plot estimated transfer function (time-domain)
        def select_samples(q):
            return q[:len(h)]

        H_hat_time = np.fft.irfft(H_hat, n=nfft, axis=0)
        to_plot = [select_samples(H_hat_time[:, idx]) for idx in indices_h_estimators]

        fig, axes = plt.subplots(1 + len(indices_h_estimators), 1, figsize=(3.5, 5.5), sharex=True, sharey=True)
        axes[0].plot(select_samples(h))
        axes[0].set_title('True impulse response')

        for idx, (h_hat, name) in enumerate(zip(to_plot, names_h_estimators)):
            axes[idx + 1].plot(h_hat)
            axes[idx + 1].set_title(name)

        fig.tight_layout()
        for ax in axes:
            ax.grid(True)

        fig.show()


    @staticmethod
    def plot_spectrogram_and_f0(x_time_domain, fs_, hop_length_, nstft, f0s=np.empty(0), what_plot='amplitude'):
        """
        Plot spectrogram and overlay f0.
        Other options are 'rel_phase' (referred to frame start) and 'abs_phase' (referred to time origin)
        """

        X_freq_domain = librosa.stft(x_time_domain, n_fft=nstft, hop_length=hop_length_)
        num_freqs, num_frames = X_freq_domain.shape
        fig_, ax_ = plt.subplots()
        if what_plot == 'amplitude':
            D_ = librosa.amplitude_to_db(np.abs(X_freq_domain), ref=np.max)
            img_ = librosa.display.specshow(D_, x_axis='time', y_axis='log', ax=ax_, sr=fs_, hop_length=hop_length_,
                                            n_fft=nstft)
            title = "Spectrogram"
            if any(f0s):
                times_ = librosa.times_like(f0s, sr=fs_, hop_length=hop_length_)
                ax_.plot(times_, f0s, label='f0', color='cyan', linewidth=3)
                title += " and f0"
                ax_.legend(loc='upper right')

        elif what_plot == 'rel_phase':
            D_ = np.angle(X_freq_domain)
            img_ = librosa.display.specshow(D_, x_axis='time', y_axis='log', ax=ax_, sr=fs_, hop_length=hop_length_,
                                            n_fft=nstft)
            title = "Relative phase (frame start)"
        elif what_plot == 'abs_phase':
            D_ = np.angle(X_freq_domain)

            # We need to reference each phase component to the beginning
            num_freqs, num_frames = D_.shape
            for ll in range(1, num_frames):
                delay_samples = hop_length_
                for kk in range(1, num_freqs):
                    phase_correction = 2 * np.pi * kk * delay_samples / nstft
                    D_[kk, ll] = np.angle(np.exp(1j * (D_[kk, ll] - phase_correction)))

            # Wrap the phase to [-pi, pi]
            D_ = np.mod(D_ + np.pi, 2 * np.pi) - np.pi

            img_ = librosa.display.specshow(D_, x_axis='time', y_axis='log', ax=ax_, sr=fs_, hop_length=hop_length_,
                                            n_fft=nstft)
            title = "Absolute phase (time origin)"

        elif what_plot == 'phase_diff':  # instantaneous frequency
            D_ = np.angle(X_freq_domain)
            D_ = np.diff(D_, axis=1) - 2 * np.pi * np.arange(num_freqs)[:, None] * hop_length_ / nstft
            D_ = np.angle(np.exp(1j * D_))
            img_ = librosa.display.specshow(D_, x_axis='time', y_axis='log', ax=ax_, sr=fs_, hop_length=hop_length_,
                                            n_fft=nstft)
            title = "Phase diff"

        else:
            raise ValueError(f"Unknown what_plot: {what_plot}. Choose 'amplitude', 'rel_phase' or 'abs_phase'")

        fig_.colorbar(img_, ax=ax_, format="%+2.f dB")
        ax_.set(title=title)
        fig_.show()

        return D_, fig_, ax_

    @staticmethod
    def plot_spectrogram_cpx_data(t, f, Zxx, amp=(0, 1)):
        """ Plot spectrogram of complex data."""

        plt.pcolormesh(t, f, np.abs(Zxx), vmin=amp[0], vmax=amp[1], shading='gouraud')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

    @staticmethod
    def plot_psd(x_time_domain, fs_, nstft, xlim=(0, 5000), y_lim=(0.5e-3, 1), return_onesided=True, title=''):
        """Plot power spectral density."""

        f, Pxx_den = scipy.signal.welch(x_time_domain, fs_, nperseg=nstft, return_onesided=return_onesided)
        plt.semilogy(f, Pxx_den)
        plt.ylim(y_lim)
        plt.xlim(xlim)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.title(title)
        plt.show()

    @staticmethod
    def get_display_names(algo_names):

        algo_names = [algo_names] if isinstance(algo_names, str) else algo_names
        display_names = algo_names.copy()

        # Replace names with display names
        # display_names = [algo_name
        #                  .replace('CW-SV', 'Wideband (prop.)')
        #                  .replace('CRB_conditional', 'Bound cond.') for algo_name in algo_names]

        return display_names

    @staticmethod
    def get_y_label(metric_name):
        # replace matric name with y label, for readability and consistency:
        # 'MSE dB' -> 'MSE [dB]'
        metric_name = metric_name.replace('RMSE dB', 'RMSE [dB]')
        metric_name = metric_name.replace('MSE dB', 'MSE [dB]')

        return metric_name

    def plot_errors(self, x_values, errors_array, suptitle=None, title=None, metric_name='metric_name',
                        algo_names='algo_name', x_label='x_label', algo_visible=None, ylim=(None, None),
                        xscale_log=False, dpi=None, colors=None, font_size=None, legend_font_size=None,
                        legend_num_cols=None, fig_ax_tuple=(None, None), legend_positioning='within_plot'):
        """
        Plots errors_array as a function of x_values. Each subplot is a metric, each line is an algorithm.
        :param legend_positioning:
        :param fig_ax_tuple:
        :param x_values: list: x-axis values
        :param errors_array: (num_x_values, num_algorithms=num_labels, (y_mean, y_mean - y_std, y_mean + y_std))
        :param metric_name: str: metric name
        :param algo_names: list: labels inside each plot
        :param suptitle: plot title
        :param x_label: label for x-axis
        :param algo_visible: ex np.array([1,0,1]) shows first and third algorithms only
        :param ylim: tuple: (ymin, ymax)
        :param xscale_log: bool: if True, x-axis is log scale
        :param dpi: int: figure dpi
        :param colors: list: colors for each algorithm
        :param font_size: str: font size for labels
        :param legend_font_size: str: font size for legend
        :param legend_num_cols: int: number of columns for legend

        :return: fig, ax
        """

        if font_size is None:
            font_size = 'x-large'

        if legend_font_size is None:
            legend_font_size = 'large'

        if colors is None:
            colors = self.colors

        while errors_array.ndim < 3:
            errors_array = errors_array[..., np.newaxis]

        if algo_visible is not None:
            errors_array = errors_array[:, np.array(algo_visible)]
            algo_names = np.array(algo_names)[algo_visible].tolist()

        algo_names = [algo_names] if isinstance(algo_names, str) else algo_names
        algo_names = Plotter.get_display_names(algo_names)

        # preprocess values
        variation_factor_name, variation_factor_values = x_label, x_values
        x_values = np.asarray([float(x) for x in variation_factor_values])
        sorted_indices = x_values.argsort()[:len(errors_array)]
        x_values = x_values[sorted_indices]
        errors_array = errors_array[sorted_indices]

        # set up plot
        if fig_ax_tuple == (None, None):
            plot_area_size = 3.5
            fig = plt.figure(figsize=(1 + plot_area_size, 1 + 1 * plot_area_size), dpi=dpi)
            ax = fig.add_subplot(111)
        else:
            fig, ax = fig_ax_tuple

        # x axis
        num_max_x_ticks = 16
        num_x_ticks = min(num_max_x_ticks, len(x_values))

        # for metric_idx, (ax, metric_name) in enumerate(zip(axes.flat, metric_names)):
        for _ in range(1):

            # find max_y and min_y considering also the std and the possible negative values
            minus = errors_array[..., 1]
            plus = errors_array[..., 2]

            min_plus, max_plus = np.min(plus), np.max(plus)
            min_minus, max_minus = np.min(minus), np.max(minus)

            max_y = min(self.mse_db_max_error, max(max_plus, max_minus))
            min_y = max(self.mse_db_min_error, min(min_plus, min_minus))
            border_len = np.abs(max_y - min_y) / 50

            # if last dimension is not unitary, it contains (mean, standard deviation = std). Otherwise, only mean is present.
            # Plot the std as a shaded area around the mean
            if errors_array.shape[-1] > 1:
                for algo_idx, algo_name in enumerate(algo_names):
                    col = matplotlib.colors.to_rgba(colors[algo_idx])  # this is in rgb form
                    shaded_area_fill_col = (*col[:3], 0.2)
                    shaded_area_edge_col = (*col[:3], 0.6)
                    ax.fill_between(x_values, errors_array[:, algo_idx, 1], errors_array[:, algo_idx, 2],
                                        facecolor=shaded_area_fill_col, edgecolor=shaded_area_edge_col)

            # Plot the mean
            for algo_idx, algo_name in enumerate(algo_names):
                mean = errors_array[:, algo_idx, 0]
                mean = np.maximum(np.minimum(mean, max_y - border_len), min_y + border_len)
                col = colors[algo_idx]

                # line_style = 'solid'  # other lines are solid
                ax.plot(x_values, mean, c=col,
                        label=algo_name, linestyle=self.linestyles[algo_idx], linewidth=1.2,
                        marker=self.markers[algo_idx], markersize=5.2, markeredgecolor='white',
                        markerfacecolor=col, markeredgewidth=0.6)

            variation_factor_name_spaces = variation_factor_name.replace('_', ' ')

            # subplot title
            if len(variation_factor_name_spaces) > 0 and variation_factor_name_spaces[-1] not in ['s', ']', ')']:
                variation_factor_name_spaces = variation_factor_name_spaces + "s"
            if title is None:
                ax.set_title(metric_name + " for different " + variation_factor_name_spaces,
                             fontsize=font_size)
            else:
                ax.set_title(title, fontsize=font_size)

            # x,y label
            if x_label is not None:
                ax.set_xlabel(x_label, fontsize=font_size)
            else:
                ax.set_xlabel(variation_factor_name_spaces, fontsize=font_size)

            ax.set_ylabel(Plotter.get_y_label(metric_name), fontsize=font_size)

            # y limits and scale
            if ylim is not None and ylim != (None, None):
                ax.set_ylim(ylim)
            elif metric_name == 'Hermitian angle' or metric_name == 'MSE dB':
                ax.set_ylim(bottom=min_y, top=max_y)

            if metric_name == 'MSE':
                ax.set_yscale("log")

            # x scale
            if xscale_log or \
                    variation_factor_name == 'noise_estimate_perturbation_amount' \
                    or variation_factor_name == 'duration_output_sec' \
                    or variation_factor_name.lower() == 'nfft' \
                    or variation_factor_name == 'duration_output_frames' \
                    or variation_factor_name == 'Time frames':
                ax.set_xscale("log")

            # x, y ticks and grid
            if ax.get_xscale() == 'log' or len(x_values) <= 4:
                # x_locator = ticker.LogLocator(base=10, numticks=num_x_ticks)
                x_minor_locator = None
                y_minor_locator = ticker.AutoMinorLocator(4)
                # if any(np.log(x_values) % 1 == 0):
                x_locator = ticker.FixedLocator(x_values)
                # x_locator = None
                formatter = ScalarFormatter()
                formatter.set_scientific(False)
                ax.xaxis.set_major_formatter(formatter) # no scientific notation in x axis
                ax.xaxis.set_minor_formatter(NullFormatter()) # no minor ticks in x axis
                # x_ticks = np.log(x_values)
            else:
                x_locator = ticker.MaxNLocator(num_x_ticks, integer=True)
                x_minor_locator = ticker.AutoMinorLocator(4)
                y_minor_locator = ticker.AutoMinorLocator(2)
                x_ticks = x_values

            # ax.set_xticks(x_ticks)
            if x_locator is not None:
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

            # improve legend
            # if legend_num_cols is None:
            # num_cols = int(np.ceil(len(algo_names) / 3)) # legend has at most 3 entries per each column.
            # legend_num_cols = 2 if len(algo_names) > 2 else 1
            handles, labels = ax.get_legend_handles_labels()
            if legend_positioning == 'within_plot':
                ax.legend(handles, labels, fontsize=legend_font_size, ncol=legend_num_cols)
            elif legend_positioning == 'outside_plot':
                # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot/43439132#43439132
                ax.legend(handles, labels, loc='lower left', fontsize='large',
                          bbox_to_anchor=(0., 1.01),
                          mode="expand", borderaxespad=0, ncol=2, frameon=False)
            else:
                pass
                # if ax.get_legend():
                #     ax.get_legend().remove()
                #     ax.legend().set_visible(False)

        if suptitle is not None:
            fig.suptitle(suptitle, fontsize=font_size)

        # if 'error_vs_variation_param' in self.which_plots:
        #     fig.tight_layout()
        #     fig.show()

        if self.save_figures:
            u.savefig(fig, folder / f'error_vs_variation_param_{metric_name}.pdf')

        return fig

    def plot_2d_scf(self, spectral_correlation_functions, f0=None, figsize=None, title=None):

        if self.sig_props is None or not self.sig_props:
            raise ValueError('sig_props must be provided')

        f_max_bin = self.sig_props['f_max_bin']
        simulated_signal = self.sig_props['simulated_signal']

        f0_title = f'{f0:.0f}' if f0 is not None else ''
        synth_or_real = 'synthetic' if simulated_signal else 'real'

        title_2d_3d = f'Estimated SCF for f0={f0_title} Hz, {synth_or_real} signal'

        hz_or_khz = 'khz'

        f2ds = []
        for scf_name in spectral_correlation_functions.keys():

            scf = spectral_correlation_functions[scf_name]['scf']
            alphas = spectral_correlation_functions[scf_name]['alphas']
            freqs = spectral_correlation_functions[scf_name]['freqs'][:f_max_bin]

            if scf_name != 'sample_cov':
                xy_label_2d = ('Cyclic frequency', 'Frequency')
                scf_plot = np.abs(scf[:f_max_bin])
            else:
                xy_label_2d = ('Frequency 1', 'Frequency 2')
                scf_plot = np.abs(scf[:f_max_bin, :f_max_bin])
                alphas = freqs

            # Add empty columns at the left to make the plot more readable. Modify alphas accordingly.
            num_columns_left_padding = 10
            scf_plot = np.hstack((np.zeros((scf_plot.shape[0], num_columns_left_padding)), scf_plot))
            alphas = np.concatenate((np.arange(-num_columns_left_padding, 0), alphas))

            if hz_or_khz.lower() == 'khz':
                freqs = freqs / 1000
                alphas = alphas / 1000
                unit = 'kHz'
            else:
                unit = 'Hz'
            xy_label_2d = (xy_label_2d[0] + f' [{unit}]', xy_label_2d[1] + f' [{unit}]')

            if title is None:
                title = title_2d_3d + f", {scf_name}"

            f2d = u.plot_matrix(scf_plot,
                                title=title,
                                xy_ticks=(alphas, freqs[:f_max_bin]),
                                xy_label=xy_label_2d, log=self.log_transform_2d_plots,
                                show_figures=self.show_figures,
                                amp_range=self.amp_range, figsize=figsize)
            f2ds.append(f2d)
            if self.save_figures:
                u.savefig(f2d, folder / f'scf_2d_{f0:.0f}hz_{scf_name}_{synth_or_real}.pdf')
            plt.close(f2d)
            plt.pause(0.01)

        return f2ds, alphas, freqs[:f_max_bin]
