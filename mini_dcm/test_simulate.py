
import unittest
import numpy as np
from .simulate import simulate_dcm

class TestSimulateDCM(unittest.TestCase):

    def setUp(self):
        """Set up common parameters for the tests."""
        self.n_regions = 2
        self.T = 100
        self.A = np.array([[0.5, 0.1], [0.1, 0.5]])
        self.C = np.array([[1.0], [0.0]])
        self.u_amp = 1.0
        self.std_process_noise = 0.01
        self.std_observation_noise = 0.1
        self.random_state = 123

    def test_output_shapes_and_types(self):
        """Test if the outputs have the correct shapes and types."""
        x, y, u = simulate_dcm(
            self.A, self.C, self.T, self.u_amp,
            self.std_process_noise, self.std_observation_noise, self.random_state
        )

        # Test types
        self.assertIsInstance(x, np.ndarray, "x should be a numpy array")
        self.assertIsInstance(y, np.ndarray, "y should be a numpy array")
        self.assertIsInstance(u, np.ndarray, "u should be a numpy array")

        # Test shapes
        self.assertEqual(x.shape, (self.T, self.n_regions), "Shape of x is incorrect")
        self.assertEqual(y.shape, (self.T, self.n_regions), "Shape of y is incorrect")
        self.assertEqual(u.shape, (self.T, 1), "Shape of u is incorrect")

    def test_reproducibility_with_random_state(self):
        """Test if the simulation is reproducible with the same random seed."""
        x1, y1, u1 = simulate_dcm(
            self.A, self.C, self.T, self.u_amp,
            self.std_process_noise, self.std_observation_noise, self.random_state
        )
        x2, y2, u2 = simulate_dcm(
            self.A, self.C, self.T, self.u_amp,
            self.std_process_noise, self.std_observation_noise, self.random_state
        )

        np.testing.assert_array_equal(u1, u2, "u should be identical for the same seed")
        np.testing.assert_array_equal(x1, x2, "x should be identical for the same seed")
        np.testing.assert_array_equal(y1, y2, "y should be identical for the same seed")

    def test_zero_noise_case(self):
        """Test the simulation when both process and observation noise are zero."""
        x, y, u = simulate_dcm(
            self.A, self.C, self.T, self.u_amp,
            std_process_noise=0.0,
            std_observation_noise=0.0,
            random_state=self.random_state
        )

        # With zero observation noise, y should be identical to x
        np.testing.assert_array_equal(x, y, "With zero observation noise, x and y should be equal")

        # Check a specific step for deterministic evolution (with zero process noise)
        # x[1] = A @ x[0] + C @ u[0]. Since x[0] and u[0] are 0, x[1] should be 0.
        self.assertTrue(np.all(x[1] == 0))
        
        # At the first stimulus onset, x should be non-zero
        first_stim_t = self.T // 4
        # x at first_stim_t+1 depends on u at first_stim_t
        x_at_stim = self.A @ x[first_stim_t-1] + (self.C @ u[first_stim_t-1]).ravel()
        x_after_stim = self.A @ x[first_stim_t] + (self.C @ u[first_stim_t]).ravel()

        np.testing.assert_array_equal(x[first_stim_t], x_at_stim)
        np.testing.assert_array_equal(x[first_stim_t+1], x_after_stim)
        self.assertTrue(np.any(x[first_stim_t+1] != 0))


    def test_stimulus_creation(self):
        """Test if the input stimulus 'u' is created correctly."""
        _, _, u = simulate_dcm(
            self.A, self.C, self.T, self.u_amp,
            self.std_process_noise, self.std_observation_noise, self.random_state
        )

        start1 = self.T // 4
        end1 = self.T // 2
        start2 = 3 * self.T // 4

        # Before first pulse
        self.assertTrue(np.all(u[:start1] == 0))
        # During first pulse
        self.assertTrue(np.all(u[start1:end1] == self.u_amp))
        # Between pulses
        self.assertTrue(np.all(u[end1:start2] == 0))
        # During second pulse
        self.assertTrue(np.all(u[start2:] == self.u_amp))

if __name__ == '__main__':
    unittest.main()
