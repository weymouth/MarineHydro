import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.PanelMethod import *

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import exp1


def wave_source(x, y, xs, ys, K):
    "Source plus generated free surface waves"
    r2 = (x-xs)**2 + (y-ys)**2       # source square-distance
    m2 = (x-xs)**2 + (y+ys)**2       # mirror sink square-distance
    Z = K * (y + ys + 1j*abs(x-xs))  # wave number scaled complex vector
    eZ = np.exp(Z)                   # propagating wave potential
    fZ = np.real(eZ*exp1(Z))         # standing wave potential
    return 0.5*np.log(r2/m2) - 2j*np.pi*eZ - 2*fZ


# Geometry creation/measurement functions
def ellipse(N ,a=1, b=1, theta1=np.pi):
    "x,y arrays around an elliptical arc"
    theta = np.linspace(-np.pi, theta1, N+1) # N+1 points for N panels
    return a*np.cos(theta), b*np.sin(theta)


def area(x, y):
    "Calculate the area of a polygon from an array of points"
    return np.trapz(x, y)


def plot_flow(x, y, q, XY, G=source, args=(), size=(7,7), ymax=None):
    # Loop through segments, superimposing the velocity
    def uv(i):
        return q[i]*velocity(*XY, x[i], y[i], x[i+1], y[i+1], G, args)
    UV = sum(uv(i) for i in range(len(x)-1))

    # Create plot
    plt.figure(figsize=size)
    ax = plt.axes(); ax.set_aspect('equal', adjustable='box')

    # Plot vectors and segments
    Q = plt.quiver(*XY, *np.real(UV))
    if np.iscomplexobj(UV):
      Q._init()
      plt.quiver(*XY, *np.imag(UV), scale=Q.scale, color='g')
    plt.plot(x,y,c='b')
    plt.ylim(None,ymax)


def wave_video(x, y, q, XY, G=wave_source, args=(4,), size=(16,6)):
    # Get complex velocity
    def uv(i): return q[i]*velocity(*XY, x[i], y[i], x[i+1], y[i+1], G, args)
    UV = sum(uv(i) for i in range(len(x)-1))

    # Plot flow and segments
    fig, ax = plt.subplots(1, 1, figsize=size)
    Q = ax.quiver(*XY, *UV)#, pivot='mid')
    ax.plot(x, y, c='b')
    ax.set_ylim(None, 0.5)
    ax.set_aspect('equal', adjustable='box')
    plt.close()

    # run through a wave period
    def update_quiver(num, Q):
        Q.set_UVC(*np.real(UV*np.exp(-2j*np.pi*num/101)))
        return Q,

    # create the animation
    return FuncAnimation(fig, update_quiver, fargs=(Q,), interval=50)


# Visualization utilities
def mask_grid(x, y, mask):
    "delete mesh points where mask is true"
    def delete(a): return np.ma.masked_array(a, mask(x, y)).compressed()
    return np.array((delete(x), delete(y)))

