

import numpy as np
from matplotlib import pyplot as plt


def source(x, y, xs, ys): return 0.5 * np.log((x-xs)**2 + (y-ys)**2)


GAUSS2 = 0.5 * (1 + np.sqrt(1/3))  # gaussian-quadrature sample point


def potential(x, y, x0, y0, x1, y1, G=source, args=()):
    "Gaussian quadrature estimate of the potential influence function"
    def dG(s): return G(x, y, x0*(1-s)+x1*s, y0*(1-s)+y1*s, *args)
    h = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    return 0.5*h*sum(dG(s) for s in [GAUSS2, 1-GAUSS2])


def velocity(x, y, x0, y0, x1, y1, G=source, args=(), eps=1e-6):
    """Finite difference estimate of the velocity influence function"""
    def phi(x, y): return potential(x, y, x0, y0, x1, y1, G, args)
    return np.array(((phi(x+eps, y) - phi(x-eps, y)) / (2*eps),   # dphi/dx
                     (phi(x, y+eps) - phi(x, y-eps)) / (2*eps)))  # dphi/dy


def plot_flow(x, y, q, XY, G=source, args=(), size=(7,7), ymax=None):
    # Loop through segments, superimposing the velocity
    def uv(i): return q[i]*velocity(*XY, x[i], y[i], x[i+1], y[i+1], G, args)
    UV = sum(uv(i) for i in range(len(x)-1))

    # Create plot
    plt.figure(figsize=size)
    ax = plt.axes()
    ax.set_aspect('equal', adjustable='box')

    # Plot vectors and segments
    Q = plt.quiver(*XY, *np.real(UV))
    if np.iscomplexobj(UV):
        Q._init()
        plt.quiver(*XY, *np.imag(UV), scale=Q.scale, color='g')
    plt.plot(x, y, c='b')
    plt.ylim(None, ymax)


def ellipse(N, a=1, b=1, theta1=np.pi):
    "x,y arrays around an elliptical arc"
    theta = np.linspace(-np.pi, theta1, N+1)  # N+1 points for N panels
    return a*np.cos(theta), b*np.sin(theta)


def mask_grid(x, y, mask):
    "delete mesh points where mask is true"
    def delete(a): return np.ma.masked_array(a, mask(x,y)).compressed()
    return np.array((delete(x), delete(y)))


def properties(x0, y0, x1, y1):
    "properties of a line segment"
    sx, sy = x1-x0, y1-y0          # segment vector
    xc, yc = x0+0.5*sx, y0+0.5*sy  # segment center
    h = np.sqrt(sx**2 + sy**2)     # segment length
    nx, ny = sy/h, -sx/h           # segment unit normal
    return xc, yc, nx, ny, h


def construct_A(x, y, G=source, args=(), aii=np.pi):
    "construct the velocity influence matrix"
    # influence of panel i on the normal velocity at each panel center
    xc, yc, nx, ny, _ = properties(x[:-1], y[:-1], x[1:], y[1:])

    def u_n(i):
        u, v = velocity(xc, yc, x[i], y[i], x[i+1], y[i+1], G, args)
        return u*nx + v*ny

    # construct matrix
    A = np.array([u_n(i) for i in range(len(yc))]).T
    A += aii * np.eye(len(yc))  # add panel self-influence
    return A, nx, ny


def added_mass(x, y, G=source, args=(), rho=1):
    "Compute the added mass matrix"
    # strength due to x,y motion
    A, n1, n2 = construct_A(x, y, G, args)
    q1 = np.linalg.solve(A, n1)
    q2 = np.linalg.solve(A, n2)

    # potential due to y,z motion (times panel width)
    xc, yc, _, _, h = properties(x[:-1], y[:-1], x[1:], y[1:])
    hF = [h*potential(xc, yc, x[i], y[i], x[i+1], y[i+1], G, args)
          for i in range(len(yc))]
    phi1, phi2 = hF@q1, hF@q2  # potential influence matrix times strength

    # sum over panels
    return -rho * np.matrix([[phi1@n1,phi1@n2], [phi2@n1,phi2@n2]])
