import numpy as np
from math import *
from trackgen import *
from camera import *
from vis import *
from GP import *
from planning import *

datdict = np.load('track1.npy').item()
track   = datdict['track']

camera = FOV( rad = 7., distcent = 3 )

x0   = np.array([0.,0.])
nvec = np.array([-1.,0.])
xtraj, detCones = planTrajectory( track, camera, x0 = x0,
                                  nvec0 = nvec, sdetectmax = 10.,
                                  ds = .5, smax = 150., rmin=1.,
                                  sigmabar = 0.15, alpha = 0.05 )#1 )

ind = np.shape(xtraj)[0]-1
xplot = xtraj[ind,:]
nvec = xplot - xtraj[ind-1,:]
nvec /= np.linalg.norm(nvec)

plotDetection( xtraj[-1,:], nvec,
               xtraj[:,0],
               xtraj[:,1],
               track, camera, cameraAct=True, visHist=detCones )
