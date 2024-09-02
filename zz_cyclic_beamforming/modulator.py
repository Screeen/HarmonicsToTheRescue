import spectral_correlation_estimator as sce
import numpy as np
import utils as u

u.set_printoptions_numpy()
u.set_plot_options()


class Modulator:
    def __init__(self, max_len_samples_, fs_, alpha_vec_hz_):
        fake_in = np.ones((1, max_len_samples_))
        self.mod_matrix = sce.SpectralCorrelationEstimator.modulate_signal_all_alphas_vec_2d(y=fake_in, fs_=fs_,
                                                                                             alpha_vec_hz=alpha_vec_hz_)
        self.demod_matrix = sce.SpectralCorrelationEstimator.modulate_signal_all_alphas_vec_2d(y=fake_in, fs_=fs_,
                                                                                               alpha_vec_hz=-alpha_vec_hz_)

    def modulate(self, y):
        return y * self.mod_matrix[..., :y.shape[-1]]

    def demodulate(self, y):
        return y * self.mod_matrix[0, :, :y.shape[-1]].conj()
        # return x * self.demod_matrix[0, :, :x.shape[-1]]


if __name__ == '__main__':
    # test_modulator()
    fs = 4000.
    max_len_samples = int(fs * 0.1)
    alpha_vec_hz = np.array([-5, 0, 500])
    modulator = Modulator(max_len_samples, fs, alpha_vec_hz)
    x = 1e-6 * np.random.randn(1, max_len_samples) \
        + 0.01*np.cos(2 * np.pi * 100 * np.arange(max_len_samples) / fs) \
        + 0.1*np.cos(2 * np.pi * 600 * np.arange(max_len_samples) / fs)
    X = np.fft.fft(x)
    X = np.fft.fftshift(X, axes=-1)

    x_mod = modulator.modulate(x)
    X_mod = np.fft.fftshift(np.fft.fft(x_mod), axes=-1)

    x_demod = modulator.demodulate(x_mod)
    X_demod = np.fft.fft(x_demod)
    X_demod = np.fft.fftshift(X_demod, axes=-1)

    u.plot([np.log10(np.abs(X[0])), np.log10(np.abs(X_mod[0])), np.log10(np.abs(X_demod[0, 0])),
            np.log10(np.abs(X_demod[0, 1]))],
           titles=['X', 'X_mod', 'X_demod0', 'X_demod1'],
           subplot_height=1.5, time_axis=False)

    # Same as above, but real part
    # u.plot([X[0].real, X_mod[0].real, X_demod[0,0].real, X_demod[0,1].real],
    #        titles=['X', 'X_mod', 'X_demod0', 'X_demod1'],
    #        subplot_height=1.5, time_axis=False)

    # Same as above, but imaginary part
    # u.plot([X[0].imag, X_mod[0].imag, X_demod[0,0].imag, X_demod[0,1].imag],
    #        titles=['X', 'X_mod', 'X_demod0', 'X_demod1'],
    #        subplot_height=1.5, time_axis=False)

    # Same as above, but unwrapped phase
    u.plot([np.unwrap(np.angle(X[0])), np.unwrap(np.angle(X_mod[0])), np.unwrap(np.angle(X_demod[0, 0])),
            np.unwrap(np.angle(X_demod[0, 1]))],
           titles=['X', 'X_mod', 'X_demod0', 'X_demod1'],
           subplot_height=1.5, time_axis=False)

    # u.plot([x[0], x_demod[0, 0], x_demod[0, 1]], titles=['x', 'x_demod0', 'x_demod1'], subplot_height=1.5,
    #        time_axis=True)

    np.allclose(np.angle(X[0]), np.angle(X_demod[0, 1]))
    np.allclose(np.abs(X[0]), np.abs(X_demod[0, 1]))

    x_demod_comparison = x_demod
    differences = np.abs(x - x_demod_comparison)
    print(f"Max difference: {np.max(differences)}, at index: {np.argmax(differences)}")
    if np.max(differences) < 1e-3:
        assert np.allclose(x, x_demod_comparison)
        print('Modulator test passed.')
