import numpy as np


def solve_endpoint_gamma(elements):
    """Solve linear-vortex endpoint strengths for one or more lifting elements.

    Parameters
    ----------
    elements : array-like or sequence of array-like
        One element with shape ``(N+1, 2)`` or a sequence of such arrays.
        Each element must list ordered endpoints from one trailing-edge side
        to the other. Consecutive points define panels.

    Returns
    -------
    list[np.ndarray] or np.ndarray
        Endpoint gamma values grouped like the input. A single input array
        returns a single ``(N+1,)`` array. A sequence returns a list of arrays.

    Notes
    -----
    The freestream velocity is fixed to ``[1, 0]``. To change the effective
    angle of attack, rotate the geometry before calling this function.
    """
    element_arrays, single_input = _normalize_elements(elements)
    (
        panel_xc,
        panel_yc,
        panel_sx,
        panel_sy,
        panel_S,
        left_indices,
        element_slices,
        endpoint_count,
    ) = _build_geometry(element_arrays)

    panel_count = panel_xc.size
    tangency = np.zeros((panel_count, endpoint_count))

    for panel_index in range(panel_count):
        u_left, v_left, u_right, v_right = _panel_endpoint_velocity(
            panel_xc,
            panel_yc,
            panel_sx,
            panel_sy,
            panel_S,
            panel_index,
            panel_xc,
            panel_yc,
        )
        tangency[:, left_indices[panel_index]] += _project_tangential(u_left, v_left, panel_sx, panel_sy)
        tangency[:, left_indices[panel_index] + 1] += _project_tangential(u_right, v_right, panel_sx, panel_sy)

    kutta = np.zeros((len(element_slices), endpoint_count))
    for element_index, (first_index, end_index) in enumerate(element_slices):
        last_index = end_index - 1
        kutta[element_index, first_index] = 1.0
        kutta[element_index, last_index] = 1.0

    matrix = np.vstack((tangency, kutta))
    rhs = np.concatenate((panel_sy, np.zeros(len(element_slices))))

    gamma = np.linalg.solve(matrix, rhs)
    gamma_elements = [gamma[start:end] for start, end in element_slices]
    return gamma_elements[0] if single_input else gamma_elements


def induced_velocity(points, elements, gamma_elements):
    """Evaluate total velocity from linear endpoint vortex strengths.

    Parameters
    ----------
    points : array-like
        Evaluation point ``(2,)`` or points ``(M, 2)``.
    elements : array-like or sequence of array-like
        Geometry in the same format accepted by ``solve_endpoint_gamma``.
    gamma_elements : array-like or sequence of array-like
        Endpoint gamma values matching ``elements``.

    Returns
    -------
    np.ndarray
        Total velocity, including the fixed freestream ``[1, 0]``, with shape
        ``(2,)`` for one point or ``(M, 2)`` for many points.
    """
    point_array, single_point = _normalize_points(points)
    element_arrays, single_element = _normalize_elements(elements)
    gamma_arrays = _normalize_gamma(gamma_elements, element_arrays, single_element)
    (
        panel_xc,
        panel_yc,
        panel_sx,
        panel_sy,
        panel_S,
        left_indices,
        element_slices,
        _,
    ) = _build_geometry(element_arrays)
    gamma = np.concatenate(gamma_arrays)

    velocity = np.zeros((point_array.shape[0], 2), dtype=float)
    velocity[:, 0] = 1.0
    for panel_index in range(panel_xc.size):
        left = left_indices[panel_index]
        u_left, v_left, u_right, v_right = _panel_endpoint_velocity(
            panel_xc,
            panel_yc,
            panel_sx,
            panel_sy,
            panel_S,
            panel_index,
            point_array[:, 0],
            point_array[:, 1],
        )
        velocity[:, 0] += gamma[left] * u_left + gamma[left+1] * u_right
        velocity[:, 1] += gamma[left] * v_left + gamma[left+1] * v_right

    return velocity[0] if single_point else velocity


def _normalize_elements(elements):
    if isinstance(elements, np.ndarray):
        element_arrays = [_validate_element(elements)]
        return element_arrays, True

    try:
        element_arrays = [_validate_element(element) for element in elements]
    except TypeError as exc:
        raise TypeError("elements must be an array of points or a sequence of element arrays") from exc

    if not element_arrays:
        raise ValueError("at least one element is required")
    return element_arrays, False


def _validate_element(element):
    array = np.asarray(element, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("each element must have shape (N+1, 2)")
    if array.shape[0] < 3:
        raise ValueError("each element must contain at least three endpoints")
    lengths = np.linalg.norm(np.diff(array, axis=0), axis=1)
    if np.any(np.isclose(lengths, 0.0)):
        raise ValueError("panels must have non-zero length")
    return array


def _normalize_points(points):
    array = np.asarray(points, dtype=float)
    if array.ndim == 1:
        if array.shape[0] != 2:
            raise ValueError("a single evaluation point must have shape (2,)")
        return array.reshape(1, 2), True
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("points must have shape (2,) or (M, 2)")
    return array, False


def _normalize_gamma(gamma_elements, elements, single_element):
    if single_element:
        gamma_arrays = [np.asarray(gamma_elements, dtype=float)]
    else:
        gamma_arrays = [np.asarray(gamma, dtype=float) for gamma in gamma_elements]

    if len(gamma_arrays) != len(elements):
        raise ValueError("gamma_elements must match the number of elements")

    for gamma, element in zip(gamma_arrays, elements):
        if gamma.ndim != 1 or gamma.shape[0] != element.shape[0]:
            raise ValueError("each gamma array must have one value per endpoint")
    return gamma_arrays


def _build_geometry(elements):
    endpoint_sizes = np.array([element.shape[0] for element in elements], dtype=int)
    panel_counts = endpoint_sizes - 1
    endpoint_starts = np.concatenate(([0], np.cumsum(endpoint_sizes)[:-1]))
    endpoint_count = int(endpoint_sizes.sum())

    x0 = np.concatenate([element[:-1, 0] for element in elements])
    y0 = np.concatenate([element[:-1, 1] for element in elements])
    x1 = np.concatenate([element[1:, 0] for element in elements])
    y1 = np.concatenate([element[1:, 1] for element in elements])
    panel_xc, panel_yc, panel_sx, panel_sy, panel_S = _panel_geometry(x0, y0, x1, y1)

    left_indices = np.concatenate([start + np.arange(count) for start, count in zip(endpoint_starts, panel_counts)])
    element_slices = [(start, start + size) for start, size in zip(endpoint_starts, endpoint_sizes)]

    return (
        panel_xc,
        panel_yc,
        panel_sx,
        panel_sy,
        panel_S,
        left_indices,
        element_slices,
        endpoint_count,
    )


def _panel_geometry(x0, y0, x1, y1):
    xc = 0.5 * (x0 + x1)
    yc = 0.5 * (y0 + y1)
    dx = x1 - xc
    dy = y1 - yc
    S = np.sqrt(dx ** 2 + dy ** 2)
    sx = dx / S
    sy = dy / S
    return xc, yc, sx, sy, S


def _panel_endpoint_velocity(panel_xc, panel_yc, panel_sx, panel_sy, panel_S, panel_index, x, y):
    return _linear_panel_velocity(
        x,
        y,
        panel_xc[panel_index],
        panel_yc[panel_index],
        panel_sx[panel_index],
        panel_sy[panel_index],
        panel_S[panel_index],
    )


def _project_tangential(u, v, sx, sy):
    return -u * sy + v * sx


def _linear_panel_velocity(x, y, xc, yc, sx, sy, S):
    lr, dt, xp, yp = _transform_xy(x, y, xc, yc, sx, sy, S)
    g = (yp * lr + xp * dt) / S
    h = (xp * lr - yp * dt) / S + 2.0
    coefficient = 0.25 / np.pi
    u_left, v_left = _rotate_uv(coefficient * (g - dt), coefficient * (h - lr), sx, sy)
    u_right, v_right = _rotate_uv(coefficient * (-g - dt), coefficient * (-h - lr), sx, sy)
    return u_left, v_left, u_right, v_right


def _transform_xy(x, y, xc, yc, sx, sy, S):
    xt = x - xc
    yt = y - yc
    xp = xt * sx + yt * sy
    yp = yt * sx - xt * sy
    lr = 0.5 * np.log(((xp - S) ** 2 + yp ** 2) / ((xp + S) ** 2 + yp ** 2))
    dt = np.arctan2(yp, xp - S) - np.arctan2(yp, xp + S)
    return lr, dt, xp, yp


def _rotate_uv(up, vp, sx, sy):
    u = up * sx - vp * sy
    v = vp * sx + up * sy
    return u, v


__all__ = ["solve_endpoint_gamma", "induced_velocity"]