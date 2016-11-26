import pylab as p
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ra=np.random.random((100))
dec=np.random.random((100))
h=np.random.random((100))
z=np.random.random((100))

datamin=min(h)
datamax=max(h)
fig=p.figure()

ax3D=fig.add_subplot(111, projection='3d')
collection = ax3D.scatter(ra, dec, z, c=h, vmin=datamin, vmax=datamax, 
                          marker='o', cmap=cm.Spectral)
p.colorbar(collection)

p.title("MKW4s-Position vs Velocity")
p.show()