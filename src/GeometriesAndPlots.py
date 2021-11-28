from GreensFunction import *
import numpy as np

# Geometry creation functions

def ellipse(N,a=1,b=1,theta1=np.pi):
  "x,y arrays around an elliptical arc"
  theta = np.linspace(-np.pi,theta1,N+1) # N+1 points for N panels
  return a*np.cos(theta),b*np.sin(theta)

def circle_cat(N,g):
  ya,za = ellipse(N//2,theta1=0)
  yb,zb = ellipse(N//2,theta1=0)
  ya += 1+g
  yb -= 1+g
  return np.append(ya,yb),np.append(za,zb),[N//2]

# Visualization functions

from matplotlib import pyplot as plt

def plot_flow(x,y,q,XY,G=source,args=(),size=(7,7),ymax=None):
  # Loop through segments, superimposing the velocity
  def uv(i): return q[i]*velocity(*XY, x[i], y[i], x[i+1], y[i+1], G, args)
  UV = sum(uv(i) for i in range(len(x)-1))

  # Create plot
  plt.figure(figsize=size)
  ax=plt.axes(); ax.set_aspect('equal', adjustable='box')

  # Plot vectors and segments
  Q = plt.quiver(*XY, *np.real(UV))
  if np.iscomplexobj(UV):
    Q._init() 
    plt.quiver(*XY, *np.imag(UV), scale=Q.scale, color='g')
  plt.plot(x,y,c='b')
  plt.ylim(None,ymax)

from matplotlib.animation import FuncAnimation

def wave_video(x,y,q,XY,G=wave_source,args=(4,),size=(16,6)):
  # Get complex velocity
  def uv(i): return q[i]*velocity(*XY, x[i], y[i], x[i+1], y[i+1], G, args)
  UV = sum(uv(i) for i in range(len(x)-1))

  # Plot flow and segments
  fig, ax = plt.subplots(1,1,figsize=size)
  Q = ax.quiver(*XY, *UV)#, pivot='mid')
  ax.plot(x,y,c='b')
  ax.set_ylim(None,0.5)
  ax.set_aspect('equal', adjustable='box')
  plt.close()

  # run through a wave period
  def update_quiver(num, Q):
      Q.set_UVC(*np.real(UV*np.exp(-2j*np.pi*num/101)))
      return Q,

  # create the animation
  return FuncAnimation(fig, update_quiver, fargs=(Q,), interval=50)

# Visualization utilities

def mask_grid(x,y,mask):
  "delete mesh points where mask is true"
  def delete(a): return np.ma.masked_array(a,mask(x,y)).compressed()
  return np.array((delete(x),delete(y)))
