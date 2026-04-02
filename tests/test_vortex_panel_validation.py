import unittest

import numpy as np

from src import solve_endpoint_gamma
from vortex_test_utils import (
    cambered_thin_foil,
    lift_coefficient,
    rotate,
    symmetric_thin_foil,
    thin_airfoil_cl0,
)


class VortexPanelValidationTests(unittest.TestCase):
    def test_symmetric_lift_curve_matches_thin_airfoil_theory(self):
        element = symmetric_thin_foil()
        alphas = np.array([0.02, 0.04, 0.06, 0.08])
        lifts = []

        for alpha in alphas:
            rotated = rotate(element, -alpha)
            gamma = solve_endpoint_gamma(rotated)
            lifts.append(lift_coefficient(rotated, gamma))

        slope, intercept = np.polyfit(alphas, np.array(lifts), 1)
        self.assertAlmostEqual(slope / (2.0 * np.pi), 1.0, delta=0.03)
        self.assertAlmostEqual(intercept, 0.0, delta=2e-3)

    def test_cambered_zero_alpha_lift_matches_linear_theory(self):
        element = cambered_thin_foil()
        gamma = solve_endpoint_gamma(element)
        lift = lift_coefficient(element, gamma)
        expected = thin_airfoil_cl0()

        self.assertAlmostEqual(lift / expected, 1.0, delta=0.03)


if __name__ == "__main__":
    unittest.main()