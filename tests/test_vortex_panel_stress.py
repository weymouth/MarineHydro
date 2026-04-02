import unittest

import numpy as np

from src import induced_velocity, solve_endpoint_gamma
from vortex_test_utils import single_element_geometry


def _outside_panel_samples(element, panel_ids=(0, 1), sample_count=80, offset=1e-3):
    xi = np.linspace(0.05, 0.95, sample_count)
    points = []
    tangents = []
    for panel_id in panel_ids:
        p0 = element[panel_id]
        p1 = element[panel_id + 1]
        segment = p1 - p0
        length = np.linalg.norm(segment)
        tangent = segment / length
        normal = np.array([-tangent[1], tangent[0]])

        for s in xi:
            points.append(p0 + s * segment + offset * normal)
            tangents.append(tangent)

    return np.asarray(points), np.asarray(tangents), sample_count


class VortexPanelStressTests(unittest.TestCase):
    def test_tangential_velocity_is_reasonably_smooth_across_two_panels(self):
        element = single_element_geometry()
        gamma = solve_endpoint_gamma(element)

        points, tangents, samples_per_panel = _outside_panel_samples(element)
        velocity = induced_velocity(points, element, gamma)
        u_tangent = np.sum(velocity * tangents, axis=1)

        relative_jumps = np.abs(np.diff(u_tangent)) / np.maximum(np.abs(u_tangent[:-1]), 1e-6)
        seam_jump = abs(u_tangent[samples_per_panel] - u_tangent[samples_per_panel - 1])

        self.assertLess(relative_jumps.max(), 0.25)
        self.assertLess(seam_jump, 0.10)


if __name__ == "__main__":
    unittest.main()