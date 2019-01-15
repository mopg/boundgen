import numpy as np
from GP import *
from camera import *

def planTrajectory( track, camera, x0 = np.array([0.,0.]),
                    nvec0 = np.array([1.,0.]),
                    ds = 0.1, dscone = 1., smax = 1e9,
                    sigma = 1., l = 1. ):

    x    = x0.copy()
    nvec = nvec0.copy()

    stot = 0.
    idc  = 0

    # initialize cone array
    ncones = len(track.xc1) + len(track.xc2)
    xcones = np.vstack( ( np.hstack( (track.xc1, track.xc2) ),
                          np.hstack( (track.yc1, track.yc2) ) ) )
    xcones = xcones.T
    ycones = np.ones( (ncones,) )

    detCones = np.zeros( (ncones,) , dtype=bool )

    # Always start with the two cones next to the starting point (get those for free)
    detCones[-1] = True
    detCones[len(track.xc1)-1] = True

    xmidpts = x.copy()
    ymidpts = np.array( [0.] )

    xtraj = np.zeros( (0,2) )

    while ( (( np.linalg.norm( x - x0 ) > track.width) or ( stot < 5. )) and (stot < smax) ):

        # detect cones
        ndetprev = np.sum( detCones )
        camera.checkVisibleMem( xcones, nvec, x, detCones )
        ndetnew = np.sum( detCones )

        # train GP (only if number of detected cones)
        if (ndetprev != ndetnew):
            xt = np.vstack( (xcones[detCones,:], xmidpts) )
            yt = np.hstack( (ycones[detCones],   ymidpts) )

            # note that this GP has a prior of mean 0. (meaning we can drive everywhere, which is true)
            gp = GP( xtrain = xt, ytrain = yt,
                     sigma = sigma, l = l )

        # evaluate cone around car and follow that path which is closest to 1.
        # TODO should be able to do this with gradient as well?
        neval = 61
        psi   = np.linspace(-12.5,12.5,neval) * pi / 180.
        yvals = np.zeros( (neval,) )
        for jj in range(0,neval):
            Tmat = np.array( ( [cos(psi[jj]),-sin(psi[jj])],
                               [sin(psi[jj]), cos(psi[jj])] ) )
            xtemp  = x + ds * np.matmul( Tmat, nvec )
            xtemp2 = np.array( [xtemp] )

            # evaluate GP
            yvals[jj] = gp.evalPosteriorMean( xtemp2 )

        print(yvals)

        yvals += 0.05 * psi**2 / (12.5*pi/180.)**2

        imax = np.argmin( yvals )

        # update position and heading direction
        psimax = psi[imax]
        Tmat   = np.array( ( [cos(psimax),-sin(psimax)],
                             [sin(psimax), cos(psimax)] ) )
        nvec = np.matmul( Tmat, nvec )
        x   += ds * nvec

        # count total travelled distance
        stot += ds

        # add to trajectory
        xtraj = np.vstack( (xtraj, x) )

        # drop a new cone if exceeding multiple of dscone
        if ( floor( stot/dscone ) > idc ):
            idc += 1
            xmidpts = np.vstack( (xmidpts, x)  )
            ymidpts = np.hstack( (ymidpts, 0.) )

        print("step")
        print(xmidpts)

    return xtraj, detCones
