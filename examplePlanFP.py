import numpy as np
from math import *
from trackgen import *
from camera import *
from vis import *
from planning import *
from falsepositives import *

# load the previously generated track (using exampleTrack.py)
datdict = np.load('track1.npy').item()
track   = datdict['track']

camera = FOV( rad = 7., distcent = 3 )

xfc, yfc, rfc = genFalsePositives( track, sigmadist = 2.0, percFP=0.3, seed=101 )

x0   = np.array([0.,0.])
nvec = np.array([-1.,0.])
xtraj, detCones = planTrajectory( track, camera, x0 = x0,
                                  nvec0 = nvec, sdetectmax = 15.,
                                  ds = .5, smax = track.length,
                                  rmin=2.5, sigmabar = 0.1,
                                  alpha = 0.15, Pcolcorr=0.51,
                                  xfcones=xfc, yfcones=yfc, rfcones=rfc )

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
animateDetection( xtraj[:,0], xtraj[:,1], track, camera, output=True,
                  xfcones=xfc, yfcones=yfc, rfcones=rfc, filename="img/track1_lowFP.gif", dpi=100 )
