import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.PanelMethod import *

import numpy as np
from scipy.special import exp1

def wave_source(x,y,xs,ys,K):
  "Source plus generated free surface waves"
  r2 = (x-xs)**2+(y-ys)**2   # source square-distance
  m2 = (x-xs)**2+(y+ys)**2   # mirror sink square-distance
  Z = K*(y+ys+1j*abs(x-xs))  # wave number scaled complex vector
  eZ = np.exp(Z)             # propagating wave potential
  fZ = np.real(eZ*exp1(Z))   # standing wave potential
  return 0.5*np.log(r2/m2)-2j*np.pi*eZ-2*fZ

from matplotlib.animation import FuncAnimation

def wave_video(x,y,q,XY,G=wave_source,args=(4,),size=(16,6),ymax=0.5):
  "Animate the induced flow over a cycle of motion"
  # Get complex velocity
  def uv(i): return q[i]*velocity(*XY, x[i], y[i], x[i+1], y[i+1], G, args)
  UV = sum(uv(i) for i in range(len(x)-1))

  # Plot flow and segments
  fig, ax = plt.subplots(1,1,figsize=size)
  Q = ax.quiver(*XY, *UV)#, pivot='mid')
  ax.plot(x,y,c='b')
  ax.set_ylim(None,ymax)
  ax.set_aspect('equal', adjustable='box')
  plt.close()

  # run through a wave period
  def update_quiver(num, Q):
      Q.set_UVC(*np.real(UV*np.exp(-2j*np.pi*num/101)))
      return Q,

  # create the animation
  return FuncAnimation(fig, update_quiver, fargs=(Q,), interval=50)

def flow_video(x,y,q,XY,G=source,args=(),size=(12,8),ymax=None):
  return wave_video(x,y,q,XY,G,args,size,ymax)