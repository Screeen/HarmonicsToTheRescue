from pathlib import Path
import librosa
import numpy as np
import scipy
from matplotlib import pyplot as plt
import pystoi
import pypesq
from tqdm import trange

# Add the path to the cyclic beamforming package
module_parent = Path(__file__).resolve().parent
import utils as u
import manager
import zz_plot_real_synthetic_vowel_helper as helper
import modulator

u.set_printoptions_numpy()

sum_noise_for_cov_stft = None
L_num_frames_cov = None
sum_noise_for_cov_mod_stft_3d = None
sum_noise_for_cov_mod_stft_3d_conj = None
alpha_vec_hz = np.array([0])


def normalize_and_pad(x, pad_to_len_):
    x = u.normalize_volume(x)
    x = m.pad_last_dim(x, pad_to_len_)
    return x


def extract_block_diagonal(arr, blk_size: int):
    arr_w, arr_h = arr.shape[-2:]
    if arr_w != arr_h:
        raise ValueError('The array is not square')

    temp = [arr[i * blk_size:(i + 1) * blk_size, i * blk_size:(i + 1) * blk_size] for i in range(arr.shape[0] // blk_size)]
    return scipy.linalg.block_diag(*temp)


if __name__ == '__main__':
    # Configs
    fs = 16000
    M = 4  # Number of microphones

    win_name = 'hann'
    dft_props = {'nfft': 512, 'fs': fs}
    dft_props['nw'] = dft_props['nfft']
    dft_props['noverlap'] = int(np.ceil(2 * dft_props['nw'] / 3))

    narrowband_mvdr = True
    wideband_mvdr = True
    play_signals = False
    real_speech = True
    minimize_noisy_cov = False
    plot_waveforms = True
    plot_spectrograms = True
    use_real_dft = False

    minimize_noise_cov = not minimize_noisy_cov

    loading = 1e-12
    target_duration = int(.6 * fs)
    noise_est_duration = int(2 * fs)
    rir_len = int(1. * fs)
    snr_db_dir_noise = 0
    snr_db_self_noise = 40
    sig_prop = {'f_max_hz': 8000, 'alpha_max_hz': 1500}
    K_nfft = dft_props['nfft'] // 2 + 1 if use_real_dft else dft_props['nfft']
    max_offset_interferer = 1

    # Paths
    m = manager.Manager()
    project_path = Path(__file__).resolve().parent.parent
    datasets_path = project_path.parent / 'datasets'
    # speech_dataset_path = datasets_path / 'north_texas_vowels' / 'data'
    # speech_dataset_path = datasets_path / 'audio'
    speech_dataset_path = datasets_path / 'anechoic'
    # speech_file_name = 'e_i.wav'
    # speech_file_name = 'kadpal05.wav'
    # speech_file_name = 'kamsii09.wav'
    # speech_file_name = 'kadper08.wav'
    speech_file_name = 'SI Harvard Word Lists Male_16khz.wav'
    # speech_file_name = 'SI Harvard Word Lists Female_16khz.wav'
    # interferer_file_name = 'k3swoo02.wav'
    # interferer_file_name = 'kadpal05.wav'
    # interferer_file_name = 'kamsii09.wav'
    # interferer_file_name = 'SI Harvard Word Lists Female_16khz.wav'
    interferer_file_name = '1-18527-B-44.wav'
    # interferer_file_name = 'e.wav'

    # dir_noise_path = datasets_path / 'north_texas_vowels' / 'data' / interferer_file_name  # can also be None if WGN is desired
    # dir_noise_path = datasets_path / 'anechoic' / interferer_file_name  # can also be None if WGN is desired
    dir_noise_path = datasets_path / 'ESC-50-master' / 'audio-selected' / interferer_file_name  # can also be None if WGN is desired
    # dir_noise_path = datasets_path / 'audio' / interferer_file_name  # can also be None if WGN is desired
    # dir_noise_path = None
    speech_path = speech_dataset_path / speech_file_name
    # speech_path = None

    ir_dataset_path = datasets_path / 'Hadad_shortcut'
    speech_ir_path = ir_dataset_path / '1.wav'
    dir_noise_ir_path = ir_dataset_path / '2.wav'

    # Calculate parameters based on settings
    dft_props['r_shift_samples'] = dft_props['nw'] - dft_props['noverlap']
    dft_props['delta_f'] = dft_props['fs'] / dft_props['nfft']
    window = scipy.signal.windows.get_window(win_name, dft_props['nw'], fftbins=True)
    window = window / np.sqrt(np.sum(window ** 2))  # normalize to unit power

    # Use the newer ShortTimeFFT class to do the same operations
    stft = scipy.signal.ShortTimeFFT(hop=dft_props['noverlap'], fs=fs, win=window, fft_mode='twosided')
    if scipy.signal.check_COLA(window, dft_props['nfft'], dft_props['noverlap']):
        raise ValueError('The window does not satisfy the COLA condition')


    def local_stft(y_, real_data=False):
        return stft.stft(y_)


    def local_istft(Y_, real_data=False):
        return stft.istft(Y_)


    if rir_len > target_duration:
        rir_len = target_duration

    if noise_est_duration < target_duration:
        noise_est_duration = target_duration

    # 1. Generation of synthetic data
    if real_speech:
        if speech_path is None:
            dry_speech = u.rng.standard_normal(size=target_duration)
            dry_speech_fs = fs
        else:
            # Load the speech sample
            offset = 0.045 if speech_dataset_path.name == 'data' else 12.5
            dry_speech, dry_speech_fs = u.load_audio_file(speech_path, fs_=fs,
                                                          offset_seconds=offset, smoothing_window=True)
            if dry_speech.shape[-1] > target_duration:
                dry_speech = dry_speech[:target_duration]
            num_reps = target_duration // dry_speech.shape[-1] + 1
            dry_speech = m.repeat_along_time_axis(dry_speech, num_reps)
            if target_duration > len(dry_speech):
                dry_speech = np.pad(dry_speech, (0, target_duration - len(dry_speech)), mode='constant')

            try:
                sig_prop['f0_range'], sig_prop['f0_over_time'] = m.find_f0_from_recording(dry_speech, fs,
                                                                                          dft_props['r_shift_samples'],
                                                                                          1024)
            except librosa.util.exceptions.ParameterError:
                sig_prop['f0_range'] = (125, 127, 129)

            fundamental_freq_hz = np.mean(sig_prop['f0_range'])
            alpha_vec_hz = np.array([n * fundamental_freq_hz for n in range(100)])
            alpha_vec_hz = alpha_vec_hz[alpha_vec_hz < sig_prop['alpha_max_hz']]
            # alpha_vec_hz = -alpha_vec_hz
            # alpha_vec_hz = np.insert(alpha_vec_hz, -1, -alpha_vec_hz)
            alpha_vec_hz = np.unique(alpha_vec_hz)

            # sig_prop['num_harmonics'] = int(np.ceil(sig_prop['f_max_hz'] / sig_prop['f0_range'][1]))
            # delta_alpha = (fs / target_duration) * 3
            # harmonic_bins = m.calculate_harmonic_frequencies(sig_prop['f0_range'], sig_prop['num_harmonics'],
            #                                                  freq_resolution=delta_alpha,
            #                                                  max_frequency=sig_prop['alpha_max_hz'])
            # alpha_vec_hz = harmonic_bins * delta_alpha
    else:
        sin_gen = helper.SinusoidGenerator()
        fundamental_freq_hz = 500
        num_harmonics = 5
        freqs_hz = np.array([fundamental_freq_hz * n for n in range(1, num_harmonics + 1)])
        win_len = target_duration
        y_a, _ = sin_gen.generate_two_harmonic_processes(freqs_hz=freqs_hz, Nw_win_length=win_len, L_num_frames=3,
                                                         fs=fs)
        y_a = helper.normalize(y_a)
        dry_speech = y_a
        dry_speech_fs = fs
        alpha_vec_hz = np.insert(freqs_hz, 0, 0)

    dry_speech = dry_speech[:target_duration]
    num_mods = len(alpha_vec_hz)
    print(f'{alpha_vec_hz = }')

    # Load the impulse responses
    speech_ir, speech_ir_fs = librosa.load(speech_ir_path, sr=fs, mono=False)
    speech_ir = speech_ir[:M, :rir_len]

    # Check if the sampling rates are the same
    if dry_speech_fs != speech_ir_fs:
        raise ValueError(f"Sampling rates are different: {dry_speech_fs = }, {speech_ir_fs = }")

    dir_noise_ir, dir_noise_ir_fs = librosa.load(dir_noise_ir_path, sr=fs, mono=False)
    dir_noise_ir = dir_noise_ir[:M, :rir_len]

    normalization = np.sum(speech_ir ** 2)
    speech_ir = speech_ir / normalization  # Normalize the impulse response to unit energy
    dir_noise_ir = dir_noise_ir / normalization  # Normalize the impulse response to unit energy

    # Convolve the speech with the impulse response
    wet = scipy.signal.convolve(dry_speech[np.newaxis, :], speech_ir, mode='full')[:, :len(dry_speech)]

    # Add noise to the signal. Two different realizations for the noise.
    # Realization 1: used for the mix
    dir_noise_wet_mix = m.get_realization_directional_noise(impulse_response=dir_noise_ir, wet_speech=wet,
                                                            snr_db=snr_db_dir_noise, sample_path=dir_noise_path, fs=fs,
                                                            offset_max=max_offset_interferer)

    # Self noise (microphone noise)
    _, self_noise_for_mix = m.add_noise_snr(wet, snr_db=snr_db_self_noise, fs=fs)
    sum_noise_mix = dir_noise_wet_mix + self_noise_for_mix

    # Mix the signals
    noisy = wet + sum_noise_mix

    # Realization 2: used for the covariance estimation
    # Directional noise
    if wet.shape[-1] < noise_est_duration:
        wet_padded = np.concatenate([wet, np.zeros((wet.shape[0], noise_est_duration - wet.shape[1]))], axis=-1)
    else:
        wet_padded = wet

    # Compute the frequency domain signals
    noisy_speech_stft = local_stft(noisy, real_data=use_real_dft)
    L_num_frames = noisy_speech_stft.shape[-1]

    # Estimate modulated signals.
    # First, prepare the modulation matrix
    max_len = np.max(np.c_[wet.shape[-1], wet_padded.shape[-1]]) + dft_props['nfft']
    mod = modulator.Modulator(max_len, fs, alpha_vec_hz)

    if 1:
        dir_noise_wet_for_cov = m.get_realization_directional_noise(impulse_response=dir_noise_ir,
                                                                    wet_speech=wet_padded,
                                                                    snr_db=snr_db_dir_noise, sample_path=dir_noise_path,
                                                                    fs=fs, offset_max=max_offset_interferer)
        dir_noise_wet_for_cov = dir_noise_wet_for_cov[:, :noise_est_duration]

        # Self noise (microphone noise)
        _, self_noise_for_for_cov = m.add_noise_snr(wet_padded, snr_db=snr_db_self_noise, fs=fs)

        # Mix the noise signals (for the covariance estimation)
        sum_noise_for_cov = dir_noise_wet_for_cov + self_noise_for_for_cov
        sum_noise_for_cov_stft = local_stft(sum_noise_for_cov, real_data=use_real_dft)
        L_num_frames_cov = sum_noise_for_cov_stft.shape[-1]

        # sum_noise_for_cov_mod_stft.shape = (M, num_modulations, dft_props['nfft'], L_num_frames)
        sum_noise_for_cov_mod = mod.modulate(sum_noise_for_cov[:, np.newaxis, :])
        sum_noise_for_cov_mod_stft = local_stft(sum_noise_for_cov_mod, real_data=False)
        sum_noise_for_cov_mod_stft_3d = np.reshape(sum_noise_for_cov_mod_stft,
                                                   (M * num_mods, dft_props['nfft'], L_num_frames_cov), order='F')
        sum_noise_for_cov_mod_stft_3d_conj = sum_noise_for_cov_mod_stft_3d.conj()

    # Modulate the wet speech
    wet_mod = mod.modulate(wet[:, np.newaxis, :])
    wet_mod_stft = local_stft(wet_mod, real_data=False)

    # Modulate the noisy speech
    noisy_speech_mod = mod.modulate(noisy[:, np.newaxis, :])
    noisy_speech_mod_stft = local_stft(noisy_speech_mod, real_data=False)

    # Modulate the impulse response
    # speech_ir = speech_ir + 1e-2 * u.rng.standard_normal(size=speech_ir.shape)
    speech_atf = np.fft.rfft(speech_ir, n=dft_props['nfft']) if use_real_dft else np.fft.fft(speech_ir,
                                                                                             n=dft_props['nfft'],
                                                                                             axis=-1)
    speech_rtf = speech_atf / speech_atf[0, :]
    # speech_rtf = speech_atf

    # speech_rtf_mod.shape = (M, num_modulations, dft_props['nfft'])
    speech_ir_mod = mod.modulate(speech_ir[:, np.newaxis, :])
    speech_atf_mod = np.fft.fft(speech_ir_mod, n=dft_props['nfft'], axis=-1)
    speech_rtf_mod = speech_atf_mod / speech_atf_mod[0, :]
    # speech_rtf_mod = speech_atf_mod

    # Reshaping
    noisy_speech_mod_stft_3d = np.reshape(noisy_speech_mod_stft, (M * num_mods, dft_props['nfft'], L_num_frames),
                                          order='F')
    wet_mod_stft_3d = np.reshape(wet_mod_stft, (M * num_mods, dft_props['nfft'], L_num_frames), order='F')
    speech_rtf_mod_2d = np.reshape(speech_rtf_mod, (M * num_mods, dft_props['nfft']), order='F')

    # Precalculate the conjugate of the signals
    noisy_speech_mod_stft_3d_conj = noisy_speech_mod_stft_3d.conj()

    # Compute ideal target covariance matrix
    wet_stft = local_stft(wet_padded, real_data=use_real_dft)
    cov_wet_nb = np.zeros((K_nfft, M, M), dtype=np.complex128)
    cov_noise_nb = np.zeros_like(cov_wet_nb)
    cov_noisy_nb = np.zeros_like(cov_wet_nb)

    cov_wet_wb = np.zeros((K_nfft, M * num_mods, M * num_mods), dtype=np.complex128)
    cov_noise_wb = np.zeros((K_nfft, M * num_mods, M * num_mods), dtype=np.complex128)
    cov_noisy_wb = np.zeros_like(cov_noise_wb)

    # Compute the covariance matrices
    for kk in range(K_nfft):
        cov_wet_nb[kk] = (wet_stft[:, kk] @ wet_stft[:, kk].conj().T) / L_num_frames
        cov_noise_nb[kk] = (sum_noise_for_cov_stft[:, kk] @ sum_noise_for_cov_stft[:, kk].conj().T) / L_num_frames_cov
        cov_noise_nb[kk] = cov_noise_nb[kk] + loading * np.eye(M)

        cov_noisy_nb[kk] = (noisy_speech_stft[:, kk] @ noisy_speech_stft[:, kk].conj().T) / L_num_frames
        cov_noisy_nb[kk] = cov_noisy_nb[kk] + loading * np.eye(M)

        cov_wet_wb[kk] = (wet_mod_stft_3d[:, kk] @ wet_mod_stft_3d[:, kk].conj().T) / L_num_frames

        cov_noise_wb[kk] = (sum_noise_for_cov_mod_stft_3d[:, kk] @
                            sum_noise_for_cov_mod_stft_3d_conj[:, kk].T) / L_num_frames_cov
        cov_noise_wb[kk] = cov_noise_wb[kk] + loading * np.eye(M * num_mods)

        cov_noisy_wb[kk] = (noisy_speech_mod_stft_3d[:, kk] @ noisy_speech_mod_stft_3d_conj[:, kk].T) / L_num_frames
        cov_noisy_wb[kk] = cov_noisy_wb[kk] + loading * np.eye(M * num_mods)

    # 2. Beamforming
    mvdr_weights = np.zeros((M, K_nfft), dtype=np.complex128)
    select_ref_weights = np.zeros_like(mvdr_weights)
    select_ref_weights[0] = 1
    mvdr_weights_wb_lcmv = np.zeros((M * num_mods, K_nfft), dtype=np.complex128)
    mvdr_weights_wb_xcorr = np.zeros_like(mvdr_weights_wb_lcmv)

    beamformed_stft = np.zeros((K_nfft, L_num_frames), dtype=np.complex128)  # shape = (num_freqs, num_frames)
    beamformed_wb_stft_lcmv = np.zeros((num_mods, K_nfft, L_num_frames), dtype=np.complex128)
    beamformed_wb_stft_xcorr = np.zeros_like(beamformed_stft)

    constraint_vec = np.ones(num_mods)

    # Precompute which bins should be processed by the wideband beamformer
    # use fundamental_freq_hz and multiple of it as the frequency bins to be processed by the wideband beamformer
    freqs_hz = np.fft.rfftfreq(dft_props['nfft'], 1 / fs) if use_real_dft else np.fft.fftfreq(dft_props['nfft'], 1 / fs)
    delta_f = freqs_hz[1] - freqs_hz[0]
    bins_skipped_by_wb_beamformer = []
    max_freq_hz_wb = 2000
    for kk in range(K_nfft):
        if np.abs(freqs_hz[kk]) % fundamental_freq_hz > (delta_f / 2) or np.abs(freqs_hz[kk]) > max_freq_hz_wb:
            bins_skipped_by_wb_beamformer.append(kk)
    bins_skipped_by_wb_beamformer.append(0)
    bins_skipped_by_wb_beamformer = np.array(np.unique(bins_skipped_by_wb_beamformer))
    freqs_processed_by_wb_beamformer = freqs_hz[np.setdiff1d(np.arange(K_nfft), bins_skipped_by_wb_beamformer)]

    snr_wb_lcmv = np.ones((K_nfft, num_mods)) * (-np.inf)

    for kk in trange(K_nfft):
        if narrowband_mvdr:
            cov_nb_kk = cov_noisy_nb[kk] if minimize_noisy_cov else cov_noise_nb[kk]

            # compute (cov_noisy_speech_stft_mod_kk)^-1 * atf_mod_kk implicitly
            cov_nb_kk_inv_rtf = scipy.linalg.solve(cov_nb_kk, speech_rtf[:, kk], assume_a='pos')
            mvdr_weights[:, kk] = cov_nb_kk_inv_rtf / (speech_rtf[:, kk].conj().T @ cov_nb_kk_inv_rtf)

            # Apply the beamformer to the frequency domain signals
            beamformed_stft[kk] = mvdr_weights[:, kk].conj().T @ noisy_speech_stft[:, kk]

        if wideband_mvdr:

            # Skip the bins that are not multiples of the fundamental frequency.
            # # Use the narrowband beamformer for these bins.
            if kk in bins_skipped_by_wb_beamformer:
                beamformed_wb_stft_lcmv[0, kk] = beamformed_stft[kk]
                beamformed_wb_stft_lcmv[1:, kk] = 0
                beamformed_wb_stft_xcorr[kk] = beamformed_stft[kk]

                mvdr_weights_wb_xcorr[:M, kk] = mvdr_weights[:, kk]
                mvdr_weights_wb_lcmv[:M, kk] = mvdr_weights[:, kk]
                continue

            # num_mods_kk = np.sum(kk_hz >= alpha_vec_hz)
            # num_mods_times_mics_kk = num_mods_kk * M
            # num_mods_kk = len(alpha_vec_hz)
            # num_mods_times_mics_kk = num_mods_kk * M
            cov_wb_kk = cov_noisy_wb[kk] if minimize_noisy_cov else cov_noise_wb[kk]

            # if kk in bins_skipped_by_wb_beamformer:
            #     # retain block-diagonal part only (non-modulated part, M x M blocks on the diagonal)
            #     cov_wb_kk = extract_block_diagonal(cov_wb_kk, M)

            ################################
            # Option 1: LCMV beamformer
            # Constrain all the modulated RTFs separately - many constraints
            # constraint_mat is a block-matrix. Has num_modulations columns and num_modulations*M rows.
            # Each column contains speech_rtf_mod_2d[slice(M * mm, M * (mm + 1)), kk] and zeros for the rest.
            constraint_mat_kk = np.zeros((num_mods * M, num_mods), dtype=np.complex128)
            for mm in range(num_mods):
                mm_M = slice(M * mm, M * (mm + 1))
                constraint_mat_kk[mm_M, mm] = speech_rtf_mod_2d[mm_M, kk]

            assume_a = 'pos'
            cov_inv_constraints = scipy.linalg.solve(cov_wb_kk, constraint_mat_kk, assume_a=assume_a)
            term2 = scipy.linalg.solve(constraint_mat_kk.conj().T @ cov_inv_constraints, constraint_vec[:num_mods],
                                       assume_a=assume_a)
            mvdr_weights_wb_lcmv[:num_mods * M, kk] = cov_inv_constraints @ term2

            ################################
            # Option 2: Constrain only the RTF in the non-modulated part
            # constraint_vec = np.concatenate([speech_rtf[:, kk], np.zeros((M * (num_mods_kk - 1)))])
            # cov_inv_constraints = scipy.linalg.solve(cov_wb_kk, constraint_vec, assume_a='pos')
            # mvdr_weights_wb[:num_mods_times_mics_kk, kk] = cov_inv_constraints / (constraint_vec.conj().T @ cov_inv_constraints)
            ################################

            ################################

            # Compute SNR of each modulation using the LCMV beamformer
            for mm in range(num_mods):
                mm_M = slice(M * mm, M * (mm + 1))
                weights = mvdr_weights_wb_lcmv[mm_M, kk]
                snr_wb_lcmv[kk, mm] = (np.real(weights.conj().T @ cov_noisy_wb[kk, mm_M, mm_M] @ weights) /
                                       np.real(weights.conj().T @ cov_noise_wb[kk, mm_M, mm_M] @ weights))
                snr_wb_lcmv[kk, mm] = np.minimum(snr_wb_lcmv[kk, mm], 50)

            # Apply the beamformers to all the modulations. Beamformed signal has shape (num_modulations, num_freqs)
            for mm in range(num_mods):
                # slice for the microphones corresponding to the mm-th modulation
                mm_M = slice(M * mm, M * (mm + 1))
                beamformed_wb_stft_lcmv[mm, kk] = mvdr_weights_wb_lcmv[mm_M, kk].conj().T @ noisy_speech_mod_stft_3d[
                    mm_M, kk]
                trans = lambda x: x ** 2
                # trans = lambda x: np.sqrt(x)
                # trans = lambda x: x
                w = trans(snr_wb_lcmv[kk, mm]) / np.sum(trans(snr_wb_lcmv[kk]))
                beamformed_wb_stft_lcmv[mm, kk] = w * beamformed_wb_stft_lcmv[mm, kk]

                # debug
                # beamformed_wb_stft_lcmv[mm, kk] = noisy_speech_mod_stft_3d[M * mm, kk]

            # Alternative beamformer from "BLIND AND INFORMED CYCLIC ARRAY PROCESSING FOR CYCLOSTATIONARY SIGNALS"
            # Only seems to retain the noisy signal
            cross_cov_x_x0_kk = (noisy_speech_mod_stft_3d[:, kk] @
                                 noisy_speech_mod_stft_3d_conj[0, kk].T) / L_num_frames
            mvdr_weights_wb_xcorr[:, kk] = scipy.linalg.solve(cov_noisy_wb[kk], cross_cov_x_x0_kk, assume_a='pos')
            beamformed_wb_stft_xcorr[kk] = mvdr_weights_wb_xcorr[:, kk].conj().T @ noisy_speech_mod_stft_3d[:, kk]
            # beamformed_wb_stft_xcorr[kk] = noisy_speech_mod_stft_3d[0, kk]

    beamformed_time = local_istft(beamformed_stft, real_data=use_real_dft).real
    beamformed_time_wb_lcmv = local_istft(beamformed_wb_stft_lcmv, real_data=False)
    beamformed_time_wb_xcorr = local_istft(beamformed_wb_stft_xcorr, real_data=use_real_dft).real

    if plot_waveforms:
        plot_list = [wet[0], noisy[0], beamformed_time, beamformed_time_wb_xcorr, sum_noise_mix, sum_noise_for_cov[:, :target_duration]]
        plot_list = [normalize_and_pad(x, target_duration) for x in plot_list]
        u.plot(plot_list,
               titles=['Wet speech', 'Noisy speech', 'Beamformed', 'Beamformed WB'], fs=fs, subplot_height=1.2)

    # Demodulate the beamformed signal and average the estimates obtained from the different modulations
    beamformed_time_wb_lcmv_demod = mod.demodulate(beamformed_time_wb_lcmv)
    beamformed_time_wb_lcmv_avg = np.mean(beamformed_time_wb_lcmv_demod.real, axis=0)

    pad_to_len = max([x.shape[-1] for x in [noisy, wet, beamformed_time, beamformed_time_wb_lcmv_avg,
                                            beamformed_time_wb_xcorr]])
    noisy = normalize_and_pad(noisy, pad_to_len)
    wet = normalize_and_pad(wet, pad_to_len)
    beamformed_time = normalize_and_pad(beamformed_time, pad_to_len)
    beamformed_time_wb_lcmv_avg = normalize_and_pad(beamformed_time_wb_lcmv_avg, pad_to_len)
    beamformed_time_wb_xcorr = normalize_and_pad(beamformed_time_wb_xcorr, pad_to_len)

    # time_indices = slice(0, target_duration)
    # plot_list = [beamformed_time_wb_lcmv_demod.real[:], beamformed_time[np.newaxis], noisy[:1], wet[:1], ]
    # plot_list = [u.normalize_volume(x)[:, time_indices] for x in plot_list]
    # titles = ['Beamformed WB demod', 'Beamformed', 'Noisy speech', 'Wet speech', ]

    if plot_spectrograms:

        def local_trans(x):
            # highest possible positive integer is np.iinfo(np.int32).max
            # max_displayed_bin = np.iinfo(np.int32).max
            max_displayed_frequency_bin = K_nfft // 2 + 1
            # max_displayed_frequency_bin = K_nfft
            # max_displayed_frequency_bin = 150
            max_time_frames = 20
            # max_time_frames = np.iinfo(np.int32).max
            return np.abs(x)[:max_displayed_frequency_bin, :max_time_frames]

        # Prepare the matrices to plot
        wet_speech_stft = local_stft(wet[0], real_data=use_real_dft)
        beamformed_wb_demod_stft = local_stft(beamformed_time_wb_lcmv_avg, real_data=False)[..., :L_num_frames]
        matrices_to_plot = [wet_speech_stft, noisy_speech_stft[0], beamformed_stft, beamformed_wb_stft_xcorr,
                            beamformed_wb_demod_stft]
        if minimize_noise_cov:
            matrices_to_plot.append(sum_noise_for_cov_stft[0])

        # Normalize the matrices
        # max_val = np.max([np.max(np.abs(matrix_ii)) for matrix_ii in matrices_to_plot])
        # matrices_to_plot = [matrix_ii / max_val for matrix_ii in matrices_to_plot]
        matrices_to_plot = [matrix_ii / np.max(np.abs(matrix_ii)) for matrix_ii in matrices_to_plot]

        # Prepare the settings for plotting
        titles = ['Wet speech', 'Noisy speech', 'Beamformed', 'Beamformed WB (xcorr)', 'Beamformed WB (LCMV)', 'Noise']
        xy_labels = ('Time frame', 'Frequency bin')
        amp_range = (-150, 0)
        plt_sett = {'xy_label': xy_labels, 'amp_range': amp_range, 'normalized': False, 'show_figures': False}

        existing_figs = []
        for ii, matrix_ii in enumerate(matrices_to_plot):
            ff = u.plot_matrix(local_trans(matrix_ii), title=titles[ii], **plt_sett)
            existing_figs.append(ff)

        # Combine the SCFs plots into a single figure
        fig = plt.figure(figsize=(6, 7.5), layout='compressed')
        for ii, (existing_fig, title) in enumerate(zip(existing_figs, titles)):
            ax = fig.add_subplot(3, 2, ii + 1)
            _ = u.fig_to_subplot(existing_fig, title, ax, xlabel='Time frame', ylabel='Frequency bin')
        for ax in fig.get_axes():
            ax.label_outer()
        fig.show()

    volume = 0.2
    if play_signals:
        u.play(wet[0], fs, volume=volume)
        u.play(noisy[0], fs, volume=volume)
        u.play(beamformed_time, fs, volume=volume)
        u.play(beamformed_time_wb_lcmv_avg, fs, volume=volume)
        # u.play(beamformed_time_wb_xcorr, fs, volume=volume)

    # 3. Evaluation
    evaluation_names = ['Noisy', 'Beamformed', 'Beamformed WB (xcorr)', 'Beamformed WB (LCMV)']
    names_simplified = ['noisy', 'bf_nb', 'bf_wb_xcorr', 'bf_wb_lcmv']
    stoi_dict = {name: -np.inf for name in names_simplified}
    pesq_dict = {name: -np.inf for name in names_simplified}
    for idx, signal in enumerate([noisy[0], beamformed_time, beamformed_time_wb_xcorr, beamformed_time_wb_lcmv_avg]):
        if np.sum(np.abs(signal)) == 0:
            stoi = -np.inf
            pesq = -np.inf
        else:
            stoi = pystoi.stoi(wet[0], signal, fs, extended=True)
            pesq = pypesq.pesq(ref=wet[0], deg=signal, fs=fs)
        stoi_dict[names_simplified[idx]] = stoi
        pesq_dict[names_simplified[idx]] = pesq

    snr_dict = {}
    for idx, (beamformer, cov_wet, cov_noise) in enumerate(zip(
            [select_ref_weights, mvdr_weights, mvdr_weights_wb_xcorr, mvdr_weights_wb_lcmv],
            [cov_wet_nb, cov_wet_nb, cov_wet_wb, cov_wet_wb],
            [cov_noise_nb, cov_noise_nb, cov_noise_wb, cov_noise_wb])):

        snr = np.zeros(K_nfft)
        for kk in range(K_nfft):
            snr[kk] = np.real(beamformer[:, kk].conj().T @ cov_wet[kk] @ beamformer[:, kk] /
                              (beamformer[:, kk].conj().T @ cov_noise[kk] @ beamformer[:, kk]))
        snr_dict[names_simplified[idx]] = snr

    # Print results in a table
    print(f"{'Signal':<25} {'STOI':<11} {'PESQ':<11} {'SNR [dB]':<11}")
    print("-" * 60)
    for name, stoi, pesq, snr in zip(evaluation_names, stoi_dict.values(), pesq_dict.values(), snr_dict.values()):
        # add a star to best result for each metric
        stoi_star = '*' if np.abs(stoi - max(stoi_dict.values())) <= 0.01 else ''
        pesq_star = '*' if np.abs(pesq - max(pesq_dict.values())) <= 0.01 else ''
        snr_star = '*' if np.abs(np.mean(snr) - max([np.mean(snr) for snr in snr_dict.values()])) <= 0.01 else ''

        snr_mean = np.mean(snr)
        snr_mean_db = 10 * np.log10(snr_mean)
        print(f"{name:<25} {stoi_star:<1}{stoi:<10.2f} {pesq_star:<1}{pesq:<10.2f} {snr_star:<1}{snr_mean_db:<10.2f}")
