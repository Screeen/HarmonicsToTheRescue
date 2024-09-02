import os
import warnings
from pathlib import Path

import librosa
import numpy as np
import scipy

import utils as u
from utils import stft, compute_correction_term, rng, eps


# noinspection PyTupleAssignmentBalance
class Manager:

    def __init__(self):
        pass

    @staticmethod
    def next_pow_of_2(number):
        return int(2 ** np.ceil(np.log2(number)))

    @staticmethod
    def choose_loud_bins(S_pow, power_ratio_quiet_definition=1000.):
        # Return indices of bins that are loud -- i.e., have power such that:
        #   S_pow > np.max(S_pow) / power_ratio_quiet_definition
        # power_ratio_quiet_definition: number of times less power than the loudest bin to be considered quiet
        # Higher power_ratio_quiet_definition means more bins are considered loud

        loud_bins = np.where(S_pow > np.max(S_pow) / power_ratio_quiet_definition)[0]
        return loud_bins

    @staticmethod
    def choose_evaluated_bins(reference_sig, dft_properties, spectral_bins_harmonics):

        reference_sig_pow = scipy.signal.welch(reference_sig, dft_properties['fs'], nperseg=dft_properties['nw'],
                                   nfft=dft_properties['nfft'], noverlap=dft_properties['noverlap'])[1]

        loud_bins = Manager.choose_loud_bins(reference_sig_pow, power_ratio_quiet_definition=1e6)

        evaluated_bins = np.intersect1d(loud_bins, spectral_bins_harmonics)
        # ignored_bins = np.setdiff1d(np.arange(nfft // 2 + 1), evaluated_bins)

        return evaluated_bins, reference_sig_pow

    @staticmethod
    def spectral_correlation_to_bifrequency_spectrum(X):
        Y = np.zeros_like(X, complex)
        Xm = np.fliplr(X)
        for ii in range(X.shape[0]):
            Y[ii, :] = np.roll(Xm[ii], ii + 1)
        return Y

    @staticmethod
    def get_stft_phase_corrected(y, fs_, Nw_, noverlap_samples_, window, complex_stft=False, phase_correction=True):
        y_stft = stft(y, fs_=fs_, window=window, Nw_=Nw_, noverlap_samples_=noverlap_samples_, complex_stft=complex_stft)
        if phase_correction:
            phase_corr = compute_correction_term(y_stft.shape, noverlap_samples_, complex_stft=complex_stft)
            y_stft = y_stft * phase_corr

        return y_stft

    @staticmethod
    def add_noise_snr(clean, snr_db, fs=None, noise=np.zeros(()), lp_filtered_noise=False):
        # Add noise to the clean signal to achieve the desired SNR.
        # "clean" is not modified.
        # return noisy, noise

        if snr_db > 100:
            return clean, np.zeros_like(clean)

        if not noise.any():
            noise = rng.normal(size=clean.shape)
        else:
            assert noise.shape == clean.shape, "Noise and clean signal must have the same shape."

        if lp_filtered_noise:
            cutoff_freq = 1000
            b, a = scipy.signal.butter(4, cutoff_freq / (fs / 2), btype='lowpass', analog=False, output='ba')
            noise = scipy.signal.lfilter(b, a, noise)

        noise_pow = np.sum(np.abs(noise) ** 2) / len(noise)
        sig_pow = np.sum(np.abs(clean) ** 2) / len(clean)
        if np.isclose(sig_pow, 0):
            raise ValueError("Signal power is too small. Select a different segment.")

        coeff = np.sqrt((sig_pow / noise_pow) * 10 ** (-snr_db / 10))

        noise = coeff * noise
        noisy = clean + noise

        # Noise pow after rescaling, to be sure SNR is as desired
        # noise_pow = np.sum(np.abs(noise) ** 2) / len(noise)
        # snr_ = 10 * np.log10(sig_pow / noise_pow)
        # print(f"{snr_ = :.2f} dB")
        # assert np.isclose(snr_, snr_db, atol=0.1)

        return noisy, noise

    @staticmethod
    def convolution_with_system(h, s, fs, nfft, noverlap, **kwargs):

        # S = stft(s, fs_=fs, Nw_=nfft, noverlap_samples_=noverlap, padding=False)

        # Use full linear convolution, then cut y to the same length as s.
        # This gives same results as multiplying H*Y in frequency domain if h[n] = a, 0, 0, ..., 0.
        y = scipy.signal.convolve(s, h, mode='full')[:len(s)]
        Y = stft(y=y, fs_=fs, Nw_=nfft, noverlap_samples_=noverlap)

        # Approximation convolution (time) <-> point-wise multiplication (frequency)
        # holds exactly only for h[n] = a, 0, 0, ..., 0.
        # Y = H[:, np.newaxis] * S[:, :Y.shape[1]]

        return y, Y

    @staticmethod
    def load_vowel_recording(N_num_samples, fs_, offset_=0, selected_people=(), smoothing_window=True):
        """
        Load a random signal from the North Texas Vowel Database.

        References
        - Assmann, P., & Katz, W. (2005). Synthesis fidelity and time-varying spectral change in vowels. Journal of the Acoustical Society of America, 117, 886-895.
        - Katz, W., & Assmann, P. (2001). Identification of children’s and adults’ vowels: Intrinsic fundamental frequency, fundamental frequency dynamics, and presence of voicing. Journal of Phonetics, 29, 23-51.
        - Assmann, P., & Katz, W. (2000) Time-varying spectral change in the vowels of children and adults. Journal of the Acoustical Society of America, 108, 1856-1866.

        URL
        https://labs.utdallas.edu/speech-production-lab/links/utd-nt-vowel-database/

        """

        module_parent = Path(__file__).resolve().parent
        dataset_path_parent = module_parent.parent / 'datasets' / 'north_texas_vowels'

        if not dataset_path_parent.exists():
            raise ValueError(f"Path {dataset_path_parent} does not exist. Download the dataset from the URL in the docstring.")

        labels_path = dataset_path_parent / 'labels.csv'
        data_path = dataset_path_parent / 'data'

        # Load labels from CSV file
        labels = np.loadtxt(labels_path, delimiter=';', skiprows=1, dtype=str)

        # Keep only labels whose first column is a valid file name (ends with two numbers)
        labels = labels[np.array([x[-2:].isdigit() for x in labels[:, 0]])]

        # Column 0 is the file name
        # Column 1 is adult or child (adult males are labeled as '1', adult females as '2', children as '3', '4' and '5')
        # Filter to keep only children (3 4 and 5)
        if not selected_people:
            # selected_people = ['3', '4', '5']
            selected_people = ['1', '2']
            # selected_people = ['1']

        labels_filtered = labels[np.isin(labels[:, 1], selected_people)]

        # Choose random sample
        random_file_name = lambda : data_path / (labels_filtered[rng.choice(labels_filtered.shape[0]), 0] + '.wav')

        counter = 0
        file_path = data_path / random_file_name()
        while not file_path.exists() and counter < 10:
            file_path = data_path / random_file_name()
            counter += 1

        s = u.load_audio_file(file_path, fs_, N_num_samples, offset_, smoothing_window)

        return s

    @staticmethod
    def load_signal_old(N_num_samples, fs_):

        # some are from https://personal.utdallas.edu/~assmann/KIDVOW1/North_Texas_vowel_database.html
        # file_name = 'long_a.wav'
        # file_name = 'kabral05.wav'
        # file_name = 'kabrii01.wav'  # hee
        # file_name = 'kadpal07.wav'  # male head
        # file_name = 'kaksaa06.wav'  # male hod, looks good in plots, f0=102 Hz
        # file_name = 'kadlil04.wav'  # heed
        file_name = 'kamwaa10.wav'
        # file_name = 'fan_noise.wav'
        # file_name = 'male.wav'  # int(3.27 * fs) offset is nice
        # file_name = 'female.wav'  # int(3 * fs) offset is nice
        # file_name = 'e_i.wav'
        # file_name = 'e.wav'

        s, fs = librosa.load(os.path.join('audio', file_name), sr=fs_)

        if len(s) < N_num_samples:
            warnings.warn(f"Signal is too short: {len(s)} samples, but {N_num_samples} samples are needed.")
            s = np.pad(s, (0, N_num_samples - len(s)))

        # offset = int(1 * fs)
        offset = int(7.15 * fs)
        # offset = int(13.4 * fs)
        # offset = int((4.5 + 3.27 + 0.25) * fs)
        # offset = int(11.15 * fs)
        # offset = int(13.4 * fs)
        if len(s) < offset + N_num_samples:
            offset = 0
        s = s[offset:offset + N_num_samples]
        win = scipy.signal.windows.tukey(N_num_samples, alpha=0.1)
        s = s * win

        if np.max(np.abs(s)) < 1e-3:
            raise ValueError("Signal is too small. Select a different segment.")

        return s

    @staticmethod
    def remove_mean_normalize(s):
        s -= np.mean(s)
        s = s / (eps + 1.1 * np.max(np.abs(s)))
        return s

    @staticmethod
    def calculate_rmse(x, x_hat):
        # rmse_max = 5
        # rmse_real_part = np.sqrt(np.mean((x_hat.real - x.real) ** 2))
        # rmse_imag_part = np.sqrt(np.mean((x_hat.imag - x.imag) ** 2))
        #
        # rmse_real_part = np.sqrt(np.mean((np.abs(x_hat) - np.abs(x)) ** 2))
        # rmse_imag_part = np.sqrt(np.mean((np.angle(x_hat) - np.angle(x)) ** 2))

        rmse_ = np.sqrt(np.mean(np.abs(x_hat - x) ** 2))
        # rmse_ = np.minimum(rmse_, rmse_max)

        # return rmse_, rmse_real_part, rmse_imag_part
        return rmse_

    @staticmethod
    def generate_impulse_response(num_samples_h):

        # Simulate a LTI system
        h = rng.uniform(size=num_samples_h, low=-1, high=1)
        decay_win = np.exp(-10 * np.arange(num_samples_h) / num_samples_h)
        h = h * decay_win

        # Energy of h should be 1
        h = h / np.sqrt(np.sum(np.abs(h) ** 2))

        return h

    @staticmethod
    def find_f0_from_recording(s, fs, R_shift_samples, nfft):

        if nfft / fs < 512 / 48000:
            warnings.warn(f"{nfft = } / {fs = } is too small. f0 estimation will be inaccurate. ")

        f0_over_time, voiced_flag_original, voiced_probs = librosa.pyin(s, sr=fs,
                                                                        fmin=librosa.note_to_hz('C2'),
                                                                        fmax=librosa.note_to_hz('C4'),
                                                                        frame_length=nfft,
                                                                        hop_length=R_shift_samples)

        S_pow_stft = np.abs(librosa.stft(s, n_fft=nfft, hop_length=R_shift_samples)) ** 2

        # Set f0 to NaN if S_pow is too small
        S_pow_threshold = np.min(S_pow_stft) + 1e-4 * (np.max(S_pow_stft) - np.min(S_pow_stft))
        f0_over_time[np.mean(S_pow_stft, axis=0) < S_pow_threshold] = np.nan

        f0_delta = np.nanstd(f0_over_time) / 2
        finite_f0 = np.isfinite(f0_over_time)

        if np.all(voiced_flag_original == False) or np.isclose(np.sum(voiced_probs[finite_f0]), 0):
            warnings.warn("No voiced frames found. Returning None.")
            f0_range = (0, 0, 0)
        else:
            f0_mean = np.average(f0_over_time[finite_f0], weights=voiced_probs[finite_f0])
            f0_min, f0_max = f0_mean - f0_delta, f0_mean + f0_delta
            if np.abs(f0_max - f0_min) > 30:
                warnings.warn(f"Estimated f0 range is too large: {f0_min} - {f0_max} Hz. ")
            f0_range = (f0_min, f0_mean, f0_max)

        return f0_range, f0_over_time

    @staticmethod
    def compute_spectral_and_cyclic_resolutions(fs, nfft, names_scf_estimators, alphas_dirichlet, N_num_samples=None):
        """ Compute spectral and cyclic resolutions based on the SCF estimators that are used. """

        delta_f = fs / nfft
        delta_alpha_dict = dict()

        if names_scf_estimators:
            for name_scf_estimator in names_scf_estimators:
                if name_scf_estimator == 'psd' or name_scf_estimator == 'sample_cov':
                    delta_alpha_dict[name_scf_estimator] = delta_f
                elif name_scf_estimator == 'dirichlet' or name_scf_estimator == 'acp':
                    delta_alpha_dict[name_scf_estimator] = alphas_dirichlet[1] - alphas_dirichlet[0]

        if N_num_samples:
            delta_alpha_min = fs / N_num_samples  # maximum possible resolution (minimum possible delta_alpha)
            for delta_alpha in delta_alpha_dict.values():
                if delta_alpha < delta_alpha_min:
                    raise ValueError(f"{delta_alpha = } is too small: it should be at least {delta_alpha_min = }.")

        return delta_f, delta_alpha_dict

    @staticmethod
    def calculate_harmonic_frequencies(f0_range, num_harmonics, freq_resolution, max_frequency):
        """ Calculate all spectral and cyclic frequencies that correspond to harmonics of f0. """

        f0_range_bins = np.array(f0_range) / freq_resolution
        f0_range_bins = np.r_[
            int(np.floor(f0_range_bins[0])), int(np.ceil(f0_range_bins[-1]))]  # rounded below and above
        # f0_mean_bins = int(np.round(f0_range_bins[1]))

        harmonic_bins = []
        for hh in range(num_harmonics + 1):
            harmonic_bins_hh = np.arange(f0_range_bins[0] * hh, f0_range_bins[1] * hh + 1)
            # harmonic_bins_hh = np.arange(f0_mean_bins * hh - 2, f0_mean_bins * hh + 2)
            harmonic_bins.extend(harmonic_bins_hh)
        harmonic_bins = np.unique(np.array(harmonic_bins))

        # Keep only bins that are in range [0, max_frequency / freq_resolution]
        harmonic_bins = harmonic_bins[harmonic_bins >= 0]
        harmonic_bins = harmonic_bins[harmonic_bins <= max_frequency / freq_resolution]

        return harmonic_bins

    def generate_simulated_signal(self, signal_properties, fs):
        # rnd_phase_amp = (True, True)
        # s = u.generate_harmonic_signal(f0_=p['f0_range'][1], fs_=fs, L_=p['N_num_samples'], num_harmonics_=p['num_harmonics'],
        #                                frequency_error_=p['frequency_error'], rnd_phase=rnd_phase_amp[0],
        #                                rnd_amplitude=rnd_phase_amp[1])

        p = signal_properties
        freqs_hz = np.arange(start=0, step=p['f0_range'][1], stop=p['f_max_hz'])
        su = u.generate_harmonic_process(freqs_hz=freqs_hz, N_samples=p['N_num_samples'], fs=fs)
        su = Manager.remove_mean_normalize(su)

        # sc1 = u.generate_harmonic_process(freqs_hz=freqs_hz, N_samples=p['N_num_samples'], fs=fs, amplitude_correlation=0.5)
        # sc1 = SysIdentifierManager.remove_mean_normalize(sc1)

        # sc2 = u.generate_harmonic_process(freqs_hz=freqs_hz, N_samples=p['N_num_samples'], fs=fs, amplitude_correlation=0.99)
        # sc2 = SysIdentifierManager.remove_mean_normalize(sc2)

        # u.play(su, fs, volume=0.3)
        # u.play(sc1, fs, volume=0.3)
        # u.play(sc2, fs, volume=0.3)

        return su

    @staticmethod
    def define_variation_list(variation_name, configs_dict, simulated_signal, fs):

        if variation_name == 'snr':
            variations_list_ = configs_dict[variation_name]
            variations_list_display_ = configs_dict[variation_name]

        elif variation_name == 'duration':
            variations_list_ = [int(x * fs) for x in configs_dict[variation_name]]  # convert to samples
            variations_list_display_ = configs_dict[variation_name]

        elif variation_name == 'f0':
            if not simulated_signal:
                raise ValueError(f"{variation_name} variation is only valid for simulated signals")
            variations_list_ = configs_dict[variation_name]
            variations_list_display_ = configs_dict[variation_name]

        elif variation_name == 'nfft':
            variations_list_ = configs_dict[variation_name]
            variations_list_display_ = variations_list_

        else:
            raise ValueError(f"Invalid variation name {variation_name}")

        return variations_list_, variations_list_display_

    @staticmethod
    def prepare_errors_array_for_plot(errors_array, names_h_estimators, names_scf_estimators, indices_h_estimators,
                                      indices_scf_estimators):
        """ Prepare errors_array for plotting. Also prepare names_h_estimators_display."""

        num_variations, num_h_estimators, num_scf_estimators, _ = errors_array.shape

        # Reshape errors_array to (num_variations, num_h_estimators * num_scf_estimators, 3)
        # algo names are repeated for each variation, but we append to their name the variation parameter
        # so that we can plot all variations in the same figure.
        errors_array = np.reshape(errors_array, (num_variations, num_h_estimators * num_scf_estimators, 3), order='A')
        names_h_estimators_display = [f"{name_H_est} ({name_SCF_est.capitalize()})"
                                      for name_H_est in names_h_estimators for name_SCF_est in names_scf_estimators]

        # Make a mask that is False for repeated entries of Wiener for scf_estimators other than the first one.
        # Do not assume that Wiener is the first entry in names_h_estimators. E.g., if
        # names_h_estimators_display = ['Antoni (acp)', 'Antoni (sample_cov)', 'Antoni (dirichlet)',
        #                               'Wiener (acp)', 'Wiener (sample_cov)', 'Wiener (dirichlet)']
        # then mask_wiener = [True, True, True, True, False, False]
        mask_wiener = np.ones((num_h_estimators * num_scf_estimators), dtype=bool)
        for h_estimator_idx in indices_h_estimators:
            for scf_estimator_idx in indices_scf_estimators:
                if names_h_estimators[h_estimator_idx] == 'Wiener' and scf_estimator_idx > 0:
                    mask_wiener[h_estimator_idx * num_scf_estimators + scf_estimator_idx] = False

        # Use the mask to delete repeated entries of Wiener, both in names_h_estimators_display and errors_array
        names_h_estimators_display = list(np.array(names_h_estimators_display)[mask_wiener])
        errors_array = errors_array[:, mask_wiener, :]

        # Remove content within () if estimator does not contain "Cyclic". E.g., 'Wiener (acp)' -> 'Wiener'
        names_h_estimators_display = [name.split(' (')[0] if 'Cyclic' not in name
                                      else name for name in names_h_estimators_display]

        return errors_array, names_h_estimators_display

    @staticmethod
    def get_realization_directional_noise(impulse_response, wet_speech, snr_db, sample_path=None, fs=16000,
                                          wgn_scaling=1e-6, offset_max=0.1):

        if sample_path is not None:
            dir_noise_dry, dir_noise_dry_fs = u.load_audio_file(sample_path, fs_=fs, offset_seconds=u.rng.uniform(0, offset_max),
                                                                smoothing_window=True)
            # Count how many repetitions are needed to make dir_noise_dry the same length as wet_speech
            num_reps = wet_speech.shape[-1] // dir_noise_dry.shape[-1] + 1
            dir_noise_dry = Manager.repeat_along_time_axis(dir_noise_dry, num_reps=num_reps)
            dir_noise_dry = u.normalize_volume(dir_noise_dry)

            dir_noise_dry_mix = dir_noise_dry[np.newaxis] + wgn_scaling * u.rng.normal(0, 1, dir_noise_dry.shape)
        else:
            dir_noise_dry_mix = u.rng.normal(0, 1, wet_speech.shape)

        dir_noise_wet_mix = scipy.signal.convolve(dir_noise_dry_mix, impulse_response, mode='full')

        # make dir_noise_wet_mix the same shape as wet_speech
        dir_noise_wet_mix = Manager.pad_last_dim(dir_noise_wet_mix, wet_speech.shape[1])
        dir_noise_wet_mix = dir_noise_wet_mix[:wet_speech.shape[0], :wet_speech.shape[1]]

        _, dir_noise_wet_mix = Manager.add_noise_snr(wet_speech, snr_db=snr_db, fs=fs, noise=dir_noise_wet_mix,
                                                     lp_filtered_noise=False)

        return dir_noise_wet_mix

    @staticmethod
    def pad_last_dim(x, N_):
        assert x.ndim <= 2, "Only 1d and 2d arrays are supported."
        # Should work both for 1d and 2d arrays
        if N_ > x.shape[-1]:
            return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, N_ - x.shape[-1])])
        else:
            return x

    @staticmethod
    def repeat_along_time_axis(x, num_reps=3):
        return np.concatenate([x] * num_reps, axis=-1)

