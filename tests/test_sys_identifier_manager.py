import unittest
import numpy as np
from manager import Manager

class TestSysIdentifierManager(unittest.TestCase):

    def setUp(self):
        self.manager = Manager()

    def test_load_signal_with_valid_input(self):
        signal = self.manager.load_vowel_recording(1000, 44100)
        self.assertEqual(len(signal), 1000)

    def test_load_signal_with_zero_samples(self):
        with self.assertRaises(ValueError):
            self.manager.load_vowel_recording(0, 44100)

    def test_load_signal_with_negative_samples(self):
        with self.assertRaises(ValueError):
            self.manager.load_vowel_recording(-1000, 44100)

    def test_load_signal_with_invalid_fs(self):
        with self.assertRaises(ValueError):
            self.manager.load_vowel_recording(1000, -44100)

    def test_next_pow_of_2_with_power_of_2(self):
        result = self.manager.next_pow_of_2(1024)
        self.assertEqual(result, 1024)

    def test_next_pow_of_2_with_non_power_of_2(self):
        result = self.manager.next_pow_of_2(1000)
        self.assertEqual(result, 1024)

    def test_next_pow_of_2_with_negative_number(self):
        with self.assertRaises(ValueError):
            self.manager.next_pow_of_2(-1000)

    def test_choose_loud_bins_with_valid_input(self):
        S_pow = np.array([1, 2, 3, 4, 5])
        result = self.manager.choose_loud_bins(S_pow, power_ratio_quiet_definition=2)

        # Should contain bins which are louder than half of the maximum power: 3, 4, 5
        expected_result = np.array([2, 3, 4])
        self.assertTrue(np.array_equal(result, expected_result), "The result does not match the expected output.")


if __name__ == '__main__':
    unittest.main()