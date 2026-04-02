import numpy as np


def single_element_geometry():
    return np.array(
        [
            [1.0, 0.0],
            [0.3, 0.08],
            [-1.0, 0.0],
            [0.3, -0.08],
            [1.0, 0.0],
        ]
    )


def two_element_geometries():
    element1 = single_element_geometry()
    element2 = np.array(
        [
            [0.5, 0.3],
            [0.0, 0.38],
            [-0.5, 0.3],
            [0.0, 0.22],
            [0.5, 0.3],
        ]
    )
    return element1, element2


def rotate(points, angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return points @ rotation.T


def symmetric_thin_foil(panel_count=120, thickness=0.01):
    upper_x = np.linspace(1.0, 0.0, panel_count // 2 + 1)
    lower_x = np.linspace(0.0, 1.0, panel_count // 2 + 1)[1:]
    upper_y = thickness * np.sin(np.pi * upper_x)
    lower_y = -thickness * np.sin(np.pi * lower_x)
    upper = np.column_stack((upper_x, upper_y))
    lower = np.column_stack((lower_x, lower_y))
    return np.vstack((upper, lower))


def naca4_mean_camber(x, m=0.02, p=0.4):
    camber = np.empty_like(x)
    slope = np.empty_like(x)
    left = x < p
    camber[left] = m / p**2 * (2.0 * p * x[left] - x[left] ** 2)
    slope[left] = 2.0 * m / p**2 * (p - x[left])
    camber[~left] = m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x[~left] - x[~left] ** 2)
    slope[~left] = 2.0 * m / (1.0 - p) ** 2 * (p - x[~left])
    return camber, slope


def cambered_thin_foil(panel_count=120, thickness=0.01, m=0.02, p=0.4):
    upper_x = np.linspace(1.0, 0.0, panel_count // 2 + 1)
    lower_x = np.linspace(0.0, 1.0, panel_count // 2 + 1)[1:]
    upper_camber, _ = naca4_mean_camber(upper_x, m, p)
    lower_camber, _ = naca4_mean_camber(lower_x, m, p)
    upper_y = upper_camber + thickness * np.sin(np.pi * upper_x)
    lower_y = lower_camber - thickness * np.sin(np.pi * lower_x)
    upper = np.column_stack((upper_x, upper_y))
    lower = np.column_stack((lower_x, lower_y))
    return np.vstack((upper, lower))


def lift_coefficient(element, gamma):
    panel_lengths = np.linalg.norm(np.diff(element, axis=0), axis=1)
    circulation = np.sum(0.5 * (gamma[:-1] + gamma[1:]) * panel_lengths)
    chord = element[:, 0].max() - element[:, 0].min()
    return -2.0 * circulation / chord


def thin_airfoil_cl0(m=0.02, p=0.4, sample_count=20000):
    theta = np.linspace(1e-8, np.pi - 1e-8, sample_count)
    x = 0.5 * (1.0 - np.cos(theta))
    _, camber_slope = naca4_mean_camber(x, m, p)
    integral = np.trapezoid(camber_slope * (np.cos(theta) - 1.0), theta)
    return 2.0 * integral