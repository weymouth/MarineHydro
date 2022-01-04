import numpy as np

def source(x,y,xs,ys): return 0.5*np.log((x-xs)**2+(y-ys)**2)

GAUSS2 = 0.5*(1+np.sqrt(1/3)) # gaussian-quadrature sample point
def potential(x,y,x0,y0,x1,y1,G=source,args=()):
  "Gaussian quadrature estimate of the potential influence function"
  def dG(s): return G(x,y,x0*(1-s)+x1*s,y0*(1-s)+y1*s,*args)
  h = np.sqrt((x1-x0)**2+(y1-y0)**2)
  return 0.5*h*sum(dG(s) for s in [GAUSS2,1-GAUSS2])

def velocity(x,y,x0,y0,x1,y1,G=source,args=(),eps=1e-6):
  "Finite difference estimate of the velocity influence function"
  def phi(x,y): return potential(x,y,x0,y0,x1,y1,G,args)
  return np.array(((phi(x+eps,y)-phi(x-eps,y))/(2*eps),  # dphi/dx
                   (phi(x,y+eps)-phi(x,y-eps))/(2*eps))) # dphi/dy

def properties(x0, y0, x1, y1):
  "properties of a line segment"
  sx, sy = x1-x0, y1-y0         # segment vector
  xc, yc = x0+0.5*sx, y0+0.5*sy # segment center 
  h = np.sqrt(sx**2+sy**2)      # segment length
  nx, ny = sy/h, -sx/h          # segment unit normal
  return xc, yc, nx, ny, h

def construct_A(x,y,G=source,args=()):
  "construct the velocity influence matrix"
  # influence of panel i on the normal velocity at each panel center
  xc, yc, nx, ny, _ = properties(x[:-1], y[:-1], x[1:], y[1:])
  def u_n(i):
    u,v = velocity(xc,yc,x[i],y[i],x[i+1],y[i+1],G,args)
    return u*nx+v*ny

  # construct matrix
  A = np.array([u_n(i) for i in range(len(yc))]).T
  A += np.pi*np.eye(len(yc)) # add panel self-influence
  return A,nx,ny

def added_mass(y,z,G=source,args=(),rho=1):
  "Compute the added mass matrix"
  # strength due to y,z motion
  A,ny,nz = construct_A(y,z,G,args)
  qy = np.linalg.solve(A,ny)
  qz = np.linalg.solve(A,nz)

  # potential due to y,z motion (times panel width)
  yc,zc,_,_,h = properties(y[:-1], z[:-1], y[1:], z[1:])
  B = [h*potential(yc,zc,y[i],z[i],y[i+1],z[i+1],G,args) for i in range(len(yc))]
  phiy,phiz = B@qy,B@qz # multiply potential influence matrix by strength

  # sum over panels
  return -rho*np.matrix([[phiy@ny,phiy@nz],[phiz@ny,phiz@nz]])


# Geometry creation/measurement functions

def ellipse(N,a=1,b=1,theta1=np.pi):
  "x,y arrays around an elliptical arc"
  theta = np.linspace(-np.pi,theta1,N+1) # N+1 points for N panels
  return a*np.cos(theta),b*np.sin(theta)

def area(x,y):
    "Calculate the area of a polygon from an array of points"
    return np.trapz(x,y)

# Visualization functions

from matplotlib import pyplot as plt

def plot_flow(x,y,q,XY,G=source,args=(),size=(7,7),ymax=None):
  "Plot the geometry and induced velocity field"

  # Loop through segments, superimposing the velocity
  def uv(i): return q[i]*velocity(*XY, x[i], y[i], x[i+1], y[i+1], G, args)
  UV = sum(uv(i) for i in range(len(x)-1))
  UV /= np.linalg.norm(UV) # normalize

  # Create plot
  plt.figure(figsize=size)
  ax=plt.axes(); ax.set_aspect('equal', adjustable='box')

  # Plot vectors and segments
  plt.quiver(*XY, *np.real(UV), scale = 1)
  plt.quiver(*XY, *np.imag(UV), scale = 1, color='g')
  plt.plot(x,y,c='b')
  plt.ylim(None,ymax)

def mask_grid(x,y,mask):
  "delete mesh points where mask is true"
  return np.array((x[~mask(x,y)],y[~mask(x,y)]))
