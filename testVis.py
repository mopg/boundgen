import numpy as np
from math import *
from trackgen import *
from camera import *
from vis import *

datdict = np.load('track1.npy').item()
track   = datdict['track']

camera = FOV( rad = 5., distcent = 6. )

ind = 260 # 55 # 1
nvec = np.array( [ track.xmid[ind+1] - track.xmid[ind],
                   track.ymid[ind+1] - track.ymid[ind] ] )
nvec = nvec / np.linalg.norm( nvec )
xpos = np.array( [ track.xmid[ind], track.ymid[ind] ] )

plotDetection( xpos, nvec,
               track.xmid[0:ind],
               track.ymid[0:ind],
               track, camera )
