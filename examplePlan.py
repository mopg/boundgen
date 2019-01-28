import numpy as np
from math import *
from trackgen import *
from camera import *
from vis import *
from planning import *

# load the previously generated track (using exampleTrack.py)
datdict = np.load('track1.npy').item()
track   = datdict['track']

camera = FOV( rad = 7., distcent = 3 )

x0   = np.array([0.,0.])
nvec = np.array([-1.,0.])
xtraj, detCones, rCones = planTrajectory( track, camera, x0 = x0,
                                          nvec0 = nvec, sdetectmax = 15.,
                                          ds = .5, smax = track.length,
                                          rmin=2.5, sigmabar = 0.1,
                                          alpha = 0.15, Pcolcorr=0.95, seed=101 )

ind = np.shape(xtraj)[0]-1
xplot = xtraj[ind,:]
nvec = xplot - xtraj[ind-1,:]
nvec /= np.linalg.norm(nvec)

# # final figure
# plotDetection( xtraj[-1,:], nvec,
#                xtraj[:,0],
#                xtraj[:,1],
#                track, camera, cameraAct=True, visHist=detCones );

# animation
animateDetection( xtraj[:,0], xtraj[:,1], track, camera,
                  rcones=rCones, output=False )
