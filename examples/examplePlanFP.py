import numpy as np
from math import *
from boundgen import genFalsePositives, planTrajectory, animateDetection, plotDetection, FOV

# load the previously generated track (using generateExampleTrack.py)
datdict = np.load('track1.npy').item()
track   = datdict['track']

camera = FOV( rad = 7., distcent = 3 )

xfc, yfc, rfc = genFalsePositives( track, sigmadist = 2.0, percFP=0.9, seed=101 )

x0   = np.array([0.,0.])
nvec = np.array([-1.,0.])
xtraj, detCones, rCones = planTrajectory( track, camera, x0 = x0,
                                          nvec0 = nvec, sdetectmax = 15.,
                                          ds = .5, smax = track.length,
                                          rmin=2.5, sigmabar = 0.1,
                                          alpha = 0.15, Pcolcorr=0.98,
                                          xfcones=xfc, yfcones=yfc, rfcones=rfc, seed=100 )

ind = np.shape(xtraj)[0]-1
xplot = xtraj[ind,:]
nvec = xplot - xtraj[ind-1,:]
nvec /= np.linalg.norm(nvec)

# # final figure
# plotDetection( xtraj[-1,:], nvec,
#                xtraj[:,0], xtraj[:,1],
#                track, camera, cameraAct=True, visHist=detCones,
#                xfcones=xfc, yfcones=yfc, rfcones=rfc );

# animation
animateDetection( xtraj[:,0], xtraj[:,1], track, camera, output=False,
                  xfcones=xfc, yfcones=yfc, rfcones=rfc,
                  rcones=rCones )
