import numpy as np
from GP import *
from camera import *

def planTrajectory( track, camera, x0 = np.array([0.,0.]),
                    nvec0 = np.array([1.,0.]),
                    ds = 0.1, rmin=5., smax = 1e9,
                    sigmabar = 0.5, alpha = 0.5, sdetectmax = 5. ):

    x    = x0.copy()
    nvec = nvec0.copy()

    stot = 0.
    idc  = 0

    # setting up distributions
    mu = track.width/2

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

    xtraj = np.zeros( (0,2) )

    while ( (( np.linalg.norm( x - x0 ) > track.width) or ( stot < 5. )) and (stot < smax) ):

        # detect cones
        ndetprev = np.sum( detCones )
        camera.checkVisibleMem( xcones, nvec, x, detCones )
        ndetnew = np.sum( detCones )

        # unit vector perpendicular to nvec
        tvec = np.array( [-nvec[1],nvec[0]] )

        # generate paths using different inverse radii
        npts  = 25
        nns   = np.linspace( -1./rmin, 1./rmin, npts )
        probs = np.zeros( (npts,) )

        for ii in range(0,npts):

            ns = nns[ii]

            # print("ns ", nns[ii])

            if abs(ns) < 1e-5:
                r = 15. # random number -- will get fixed later
            else:
                r = 1./ns

            xcenter = x + tvec * r

            for icone in range(0,ncones):

                if not detCones[icone]:
                    continue

                xcone = xcones[icone,:]

                # check whether cone is ahead of the car, if not skip
                vtemp = xcone - x
                if np.dot( vtemp, nvec ) < 0.:
                    continue

                rcone = abs( abs(r) - np.linalg.norm( xcone - xcenter ) )

                v1 = xcone - xcenter
                v2 = x - xcenter
                theta = acos( np.dot( v1, v2 ) / ( np.linalg.norm(v1) * np.linalg.norm(v2) ) )

                s = theta * abs(r)

                if abs(ns) < 1e-5:
                    v1    = xcone - x
                    s     = np.dot( v1, nvec )
                    rcone = np.linalg.norm( v1 - np.dot(v1,nvec) * nvec )

                if s > sdetectmax:
                    continue

                # print(" icone ", icone)
                # print("     v1 ", v1)
                # print("     v2 ", v2)
                # print("     xcenter ", xcenter)
                # print("     xcone ", xcone)
                # print("     s ", s)
                # print("     rcone ", rcone)
                # print(" - (log(rcone) - log(mu))**2 ", - (log(rcone) - log(mu))**2 )


                # log likelihood
                sigma = sigmabar * exp(alpha*s)
                logprob  = - log(rcone) - log(sigma*2*pi) - (log(rcone) - log(mu))**2 / (2*sigma**2)
                probs[ii] += logprob

                prob = 1/rcone * 1/(sigma*sqrt(2*pi)) * exp( - (log(rcone) - log(mu))**2 / (2*sigma**2) )

                # print("     sigma ", sigma )
                # print("     logprob  ", logprob )
                # print("     prob ", prob)

        # find maximum likelihood
        imax = np.argmax( probs )

        nsmax = nns[imax]
        # print("     nns ", nns)
        # print("     probs ", probs)
        # print("     nsmax ", nsmax)

        # update position and heading
        theta = ds*nsmax
        Tmat  = np.array( ( [cos(theta),-sin(theta)],
                            [sin(theta), cos(theta)] ) )

        # x   += ds * nvec

        if abs(nsmax) < 1e-5:
            # straight line
            x += ds * nvec
        else:
            r = 1./nsmax
            x += nvec * r * sin(theta) + tvec * ( r - r*cos(theta) )

        # update heading
        nvec = np.matmul( Tmat, nvec )

        # count total travelled distance
        stot += ds

        # add to trajectory
        xtraj = np.vstack( (xtraj, x) )

    return xtraj, detCones
