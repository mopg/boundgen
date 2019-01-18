import numpy as np
from math import *
import matplotlib.pyplot as plt
from trackgen import *
from camera import *

def plotDetection( xpos, nvec, xtraj, ytraj, track, camera,
                   cameraAct = True,
                   visHist = np.zeros( (0, ), dtype=bool ) ):

    # plot initial track
    plt.figure()
    if track.left:
        plt.fill(track.xb1,track.yb1, '0.75' )
        plt.fill(track.xb2,track.yb2, 'w' )
    else:
        plt.fill(track.xb2,track.yb2, '0.75' )
        plt.fill(track.xb1,track.yb1, 'w' )

    plt.plot(track.xb1,track.yb1,linewidth=2,color='k')
    plt.plot(track.xb2,track.yb2,linewidth=2,color='k')

    plt.plot( track.xc1, track.yc1, 'ko', fillstyle='none' )
    plt.plot( track.xc2, track.yc2, 'ko', fillstyle='none' )

    # plot trajectory
    if len(xtraj) > 0:
        plt.plot( xtraj, ytraj, linewidth=1,color='m' )

    # compute visible cones
    xcones = np.vstack( [ np.hstack( [ track.xc1, track.xc2 ] ),
                          np.hstack( [ track.yc1, track.yc2 ] ) ] )
    xcones = xcones.T

    vis = camera.checkVisible( xcones, nvec, xpos )

    xcam, ycam = camera.getOutline( nvec, xpos )

    # finish plotting
    #   plot history of cone detections
    if (np.shape( visHist )[0] > 0):
        plt.plot( xcones[visHist,0], xcones[visHist,1], 'ko' )

    #   plot camera and active cone detection
    if cameraAct:
        plt.fill( xcam, ycam, 'g', alpha=0.25)
        plt.plot( xcones[vis,0], xcones[vis,1], 'ro' )

    # plot car
    xcar, ycar = carOutline( xcar = xpos, nvec = nvec, scl = 2. )
    plt.fill( xcar, ycar, 'k')

    plt.axis('equal')
    plt.show()

def carOutline( xcar = np.array([0.,0.]),
                scl = 1., nvec = np.array([1.,0.]) ):

    x = scl * np.array( [-0.5,-0.5,-0.25,-0.25,-0.5,-0.5,
                          0.5, 0.5, 0.25, 0.25, 0.5, 0.5] ) * 0.7
    y = scl * np.array( [ 0.5, 0.3, 0.3,-0.3,-0.3,-0.5,
                         -0.5,-0.3,-0.3, 0.3, 0.3, 0.5] )

    tvec = np.array( [nvec[1],-nvec[0]])

    xc = xcar[0] + y * nvec[0] + x * tvec[0]
    yc = xcar[1] + y * nvec[1] + x * tvec[1]

    return (xc,yc)
