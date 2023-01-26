from ubo_sampling_functions.functions import GaussianMixture
import numpy
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

gmixture = GaussianMixture()
func = gmixture.call

init = 0
end = 1
step = 0.025

x = numpy.arange(init, end, step)
y = numpy.arange(init, end, step)
X,Y = meshgrid(x, y) # grid of point

print('eval mu {}'.format(func([0.14285714285714285, 0.25])))

Z = numpy.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = [X[i,j], Y[i,j]]
        Z[i,j] = -func(point)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                      cmap=cm.RdBu,linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()