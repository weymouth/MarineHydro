import unittest

import numpy as np
from numpy.testing import assert_allclose

from src import induced_velocity, solve_endpoint_gamma
from vortex_test_utils import single_element_geometry, two_element_geometries


class VortexPanelRegressionTests(unittest.TestCase):
    def test_single_element_zero_aoa(self):
        element = single_element_geometry()
        points = np.array([[2.0, 0.0], [0.0, 0.5]])

        gamma = solve_endpoint_gamma(element)
        velocity = induced_velocity(points, element, gamma)

        expected_gamma = np.array(
            [0.422306168, -1.06803257, 0.0, 1.06803257, -0.422306168]
        )
        expected_velocity = np.array(
            [[0.994905294, 0.0], [1.0320769441, 0.0142684837]]
        )

        assert_allclose(gamma, expected_gamma, atol=1e-8, rtol=1e-8)
        self.assertAlmostEqual(float(gamma[0] + gamma[-1]), 0.0, places=12)
        assert_allclose(velocity, expected_velocity, atol=1e-8, rtol=1e-8)

    def test_two_element_zero_aoa(self):
        element1, element2 = two_element_geometries()
        point = np.array([1.5, 0.1])

        gamma1, gamma2 = solve_endpoint_gamma([element1, element2])
        velocity = induced_velocity(point, [element1, element2], [gamma1, gamma2])

        expected_gamma1 = np.array(
            [-1.08655798, -1.16675892, 0.09760162, 1.09104981, 1.08655798]
        )
        expected_gamma2 = np.array(
            [0.32474741, -1.21213898, -0.21302743, 1.23708112, -0.32474741]
        )
        expected_velocity = np.array([0.97829479, -0.00967783])

        assert_allclose(gamma1, expected_gamma1, atol=1e-8, rtol=1e-8)
        assert_allclose(gamma2, expected_gamma2, atol=1e-8, rtol=1e-8)
        self.assertAlmostEqual(float(gamma1[0] + gamma1[-1]), 0.0, places=12)
        self.assertAlmostEqual(float(gamma2[0] + gamma2[-1]), 0.0, places=12)
        assert_allclose(velocity, expected_velocity, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()