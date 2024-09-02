import unittest
from spectral_correlation_estimator import SpectralCorrelationEstimator
import numpy as np

class TestSpectralCorrelationEstimator(unittest.TestCase):

    def setUp(self):
        dft_props = {'nfft': 1024, 'fs': 44100, 'nw': 1024}
        self.spectral_correlation_estimator = SpectralCorrelationEstimator(dft_props)

    def test_modulate_signal_all_alphas(self):
        # Compare modulate_signal_all_alphas and modulate_signal_all_alphas_vec
        # def modulate_signal_all_alphas(L_, alpha_vec_hz, fs_, y):
        L_ = 1024
        alpha_vec_hz = np.array([0, 100, 200])
        fs_ = 44100
        y = np.random.rand(1024)
        result = self.spectral_correlation_estimator.modulate_signal_all_alphas(L_, alpha_vec_hz, fs_, y)
        result_vec = self.spectral_correlation_estimator.modulate_signal_all_alphas_vec(y, alpha_vec_hz, fs_, L_)
        self.assertTrue(np.allclose(result, result_vec), "The result does not match the expected output.")

