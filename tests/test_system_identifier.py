import unittest
import numpy as np
from system_identifier import SystemIdentifier
from numpy import array as arr

class SystemIdentifierTests(unittest.TestCase):

    def setUp(self):
        self.sig_props = {'f0_range': (100, 200), 'num_harmonics': 5}
        self.spectral_bins_harmonics = np.array([1, 2, 3, 4, 5])
        self.cyclic_bins_harmonics_dict = {'estimator1': arr([1, 2, 3]), 'estimator2': arr([2, 3, 4])}
        self.system_identifier = SystemIdentifier(self.sig_props, self.spectral_bins_harmonics, self.cyclic_bins_harmonics_dict)

    def test_system_identifier_run_with_valid_input(self):
        nfft = 1024
        psd_shape = (nfft // 2 + 1,)
        alphas = np.random.rand(*psd_shape)  # generate alphas with the same shape as psd
        scf_shape = (nfft // 2 + 1, len(self.cyclic_bins_harmonics_dict['estimator1']) * 2)
        S_in_in_dict = {'estimator1': {'psd': np.random.rand(*psd_shape), 'alphas': alphas, 'scf': np.random.rand(*scf_shape)},
                        'estimator2': {'psd': np.random.rand(*psd_shape), 'alphas': alphas, 'scf': np.random.rand(*scf_shape)}}
        S_out_in_dict = {'estimator1': {'psd': np.random.rand(*psd_shape), 'alphas': alphas, 'scf': np.random.rand(*scf_shape)},
                         'estimator2': {'psd': np.random.rand(*psd_shape), 'alphas': alphas, 'scf': np.random.rand(*scf_shape)}}
        names_h_estimators = ['Time-domain Wiener', 'Antoni']
        names_scf_estimators = ['estimator1', 'estimator2']
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        result = self.system_identifier.run(S_in_in_dict, S_out_in_dict, nfft, names_h_estimators, names_scf_estimators,
                                            x, y)
        self.assertEqual(result.shape, (nfft // 2 + 1, len(names_h_estimators), len(names_scf_estimators)))

    def test_system_identifier_run_with_invalid_input(self):
        S_in_in_dict = {'estimator1': {'psd': np.array([1, 2, 3])}, 'estimator2': {'psd': np.array([2, 3, 4])}}
        S_out_in_dict = {'estimator1': {'psd': np.array([1, 2, 3])}, 'estimator2': {'psd': np.array([2, 3, 4])}}
        nfft = 1024
        names_h_estimators = ['Time-domain Wiener', 'Antoni']
        names_scf_estimators = ['estimator1', 'estimator2']
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4])
        with self.assertRaises(ValueError):
            self.system_identifier.run(S_in_in_dict, S_out_in_dict, nfft, names_h_estimators, names_scf_estimators, x, y)

if __name__ == '__main__':
    unittest.main()
