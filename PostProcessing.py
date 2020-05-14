import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from matplotlib import pyplot as plt
import numpy as np
from vortexpanel import VortexPanel as vp
from vortexpanel import BoundaryLayer as bl

def C_L(panels,alpha):
    """ Calculate the coefficient of lift for a solved Panel Array

    Inputs:
    panels    -- a solved Panel Array
    alpha     -- the angle of attack

    Outputs:
    The coefficient of lift

    Example:
    ellipse = vp.make_ellipse(N=32,t_c=0.5) # make a 1:2 elliptical Panel array
    ellipse.solve_gamma_O2(alpha=0.1)       # solve the ellipse flow using a second order panel array method
    print(C_L(ellipse, alpha=0.1))          # print the coefficient of lift
    """
    gamma, xc, S, sx, sy = panels.get_array('gamma','xc','S','sx','sy') # use the solved flow to retrieve gamma, xc, S, sx, sy
    c = max(xc)-min(xc)                                                 # calculate the length of the foil chord
    perp = sx*np.cos(alpha)+sy*np.sin(alpha)                            # calculate the normal 
    return -sum((1-gamma**2)*2*S*perp)/c

def C_gamma(panels):
    top,bot=panels.split()
    _,_,iSep = top.thwaites()
    gamma_top = bl.sep(top.get_array('gamma'),iSep)
    w_top = bl.sep(top.get_array('yc'),iSep)
    y = panels.get_array('yc')
    t = max(y)-min(y)
    return 2*w_top/t * gamma_top**2     # want these things at the seperation point

alpha=np.radians(30)

ellipse = vp.make_ellipse(N=32,t_c=0.5) # make a 1:2 elliptical Panel array
ellipse.solve_gamma_O2(alpha=alpha)       # solve the ellipse flow using a second order panel array method
print(C_L(ellipse, alpha=alpha))          # print the coefficient of lift