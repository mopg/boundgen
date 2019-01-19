import numpy as np
from math import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from trackgen import *
from camera import *

def plotDetection( xpos, nvec, xtraj, ytraj, track, camera,
                   cameraAct = True,
                   visHist = np.zeros( (0, ), dtype=bool ) ):
    '''
        Plots results from planning phase.
        'xpos' is the final position for which the active cones
        and the camera outlines are plotted.
        'nvec' is the heading of the car at 'xpos'.
    '''

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
    '''
        Plots dummy car.
    '''

    x = scl * np.array( [-0.5,-0.5,-0.25,-0.25,-0.5,-0.5,
                          0.5, 0.5, 0.25, 0.25, 0.5, 0.5] ) * 0.7
    y = scl * np.array( [ 0.5, 0.3, 0.3,-0.3,-0.3,-0.5,
                         -0.5,-0.3,-0.3, 0.3, 0.3, 0.5] )

    tvec = np.array( [nvec[1],-nvec[0]])

    xc = xcar[0] + y * nvec[0] + x * tvec[0]
    yc = xcar[1] + y * nvec[1] + x * tvec[1]

    return (xc,yc)

def animateDetection( xtraj, ytraj, track, camera, output=True ):
    '''
        Animates the detection process (after the fact,
        so active cones are re-computed).
        Note that the active cones can be slightly different from what happened
        in the online planning phase, because we compute the heading of the car
        from finite differences of `xtraj` and `ytraj`.
    '''

    # initialize figure window
    if output:
        fig = plt.figure(dpi=225)
    else:
        fig = plt.figure()

    xmin = 1.1 * min( np.min(track.xc1), np.min(track.xc2) ) - 2.
    xmax = 1.1 * max( np.max(track.xc1), np.max(track.xc2) ) + 2.
    ymin = 1.1 * min( np.min(track.yc1), np.min(track.yc2) ) - 2.
    ymax = 1.1 * max( np.max(track.yc1), np.max(track.yc2) ) + 2.
    ax1 = plt.axes(xlim=(xmin, xmax), ylim=(ymin,ymax))

    lines = []

    # plot initial track
    if track.left:
        ax1.fill(track.xb1,track.yb1, '0.75' )
        ax1.fill(track.xb2,track.yb2, 'w' )
    else:
        ax1.fill(track.xb2,track.yb2, '0.75' )
        ax1.fill(track.xb1,track.yb1, 'w' )

    ax1.plot(track.xb1,track.yb1,linewidth=2,color='k')
    ax1.plot(track.xb2,track.yb2,linewidth=2,color='k')

    ax1.plot( track.xc1, track.yc1, 'ko', fillstyle='none' )
    ax1.plot( track.xc2, track.yc2, 'ko', fillstyle='none' )

    # prepare cones
    xcones = np.vstack( [ np.hstack( [ track.xc1, track.xc2 ] ),
                          np.hstack( [ track.yc1, track.yc2 ] ) ] )
    xcones = xcones.T

    vis     = np.zeros( (np.shape(xcones)[0], ), dtype=bool )
    vishist = np.zeros( (np.shape(xcones)[0], ), dtype=bool )

    # prepare plots
    #   trajectory
    trajline = ax1.plot( [], [], linewidth=1,color='m' )[0]
    lines.append(trajline)

    #   car outline
    carfill = ax1.fill( [], [], 'k')[0]
    lines.append(carfill)

    #   history cones
    histcones = ax1.plot( [], [], 'ko' )[0]
    lines.append(histcones)

    #   active cones
    actcones = ax1.plot( [], [], 'ro' )[0]
    lines.append(actcones)

    #   camera outline
    camoutline = ax1.fill( [], [], 'g', alpha=0.25)[0]
    lines.append(camoutline)

    def init():
        vishist[:] = False
        return lines

    def animate(i):

        nvec = np.array( [xtraj[i+1] - xtraj[i], ytraj[i+1] - ytraj[i]] )
        nvec /= np.linalg.norm(nvec)

        xpos = np.array([xtraj[i+1],ytraj[i+1]])

        # plot car
        xcar, ycar = carOutline( xcar = xpos, nvec = nvec, scl = 2. )

        # check visible cones
        vis = camera.checkVisible( xcones, nvec, xpos )
        xcam, ycam = camera.getOutline( nvec, xpos )

        # update plots
        lines[0].set_data( xtraj[0:i+1], ytraj[0:i+1] )
        lines[1].set_xy( np.vstack( (xcar, ycar) ).T )
        lines[2].set_data( xcones[vishist,0], xcones[vishist,1] )
        lines[3].set_data( xcones[vis,0], xcones[vis,1] )
        lines[4].set_xy( np.vstack( (xcam, ycam) ).T )

        vishist[vis] = True

        return lines

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(xtraj)-1, interval=10, blit=True)

    if output:
        anim.save("track.gif", fps=30)

    plt.show()
