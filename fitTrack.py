import numpy as np
from math import *
from trackgen import *
from GP import *
import matplotlib.pyplot as plt

datdict = np.load('track1.npy').item()
xc1 = datdict['xc1']
yc1 = datdict['yc1']
xc2 = datdict['xc2']
yc2 = datdict['yc2']

xtrain1 = np.vstack( [xc1,yc1] )
xtrain1 = xtrain1.T
xtrain2 = np.vstack( [xc2,yc2] )
xtrain2 = xtrain2.T
xtrain = np.vstack( [xtrain1,xtrain2] )
ytrain = np.hstack( [3.*np.ones( np.shape(xtrain1)[0] ),
                     3.*np.ones( np.shape(xtrain2)[0] )] )

gp = GP( sigma = 8., l = 3., xtrain=xtrain, ytrain=ytrain )

# evaluate posterior
n1d = 100
x1d = np.linspace(-10.,60.,n1d)
y1d = np.linspace(-5.,50.,n1d)

[xv,yv] = np.meshgrid(x1d,y1d)

x = np.vstack( [ np.reshape( xv, (n1d**2,) ), np.reshape( yv, (n1d**2,) ) ] )
x = x.T

(f, stdf) = gp.evalPosterior( x )

# plot
fv = np.reshape( f, (n1d,n1d) )

plt.figure()
cs = plt.contour(xv,yv,fv,[3.])
plt.axis('equal')
plt.show()
