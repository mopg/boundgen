import numpy as np
from camera import *

def planTrajectory( track, camera, x0 = np.array([0.,0.]),
                    nvec0 = np.array([1.,0.]),
                    ds = 0.1, rmin=5., smax = 1e9,
                    sigmabar = 0.5, alpha = 0.5, sdetectmax = 5.,
                    Pcolcorr = 0.7,
                    nsamples = 25,
                    xfcones = np.zeros( (0,) ),
                    yfcones = np.zeros( (0,) ),
                    rfcones = np.zeros( (0,), dtype=bool ),
                    seed=None ):

    '''
        Plans a trajectory starting from 'x0'. In this algorithm detection and
        planning happen at the same time. The car finds the 'most likely path'
        through the cones.
            - track:        track with cones
            - camera:       camera class that determines Field of View
            - x0:           initial position
            - nvec0:        initial heading
            - ds:           distance step
            - rmin:         minimum turning radius used in planning
            - smax:         maximum distance car can traverse
            - sigmabar:     standard deviation of cone position for cones next to car
            - alpha:        growth rate for standard deviation
            - sdetectmax:   maximum distance between cone and car for the cone
                            to be used in planning
            - Pcolcorr:     Probability that color detection is accurate
            - nsamples:     Number of trial trajectories for each planning step
            - xfcones:      x-coordinates of stationary false cone detections
            - yfcones:      y-coordinates of stationary false cone detections
            - rfcones:      are stationary cones detected as right (true) or left (false) cones?
            - seed:         random seed used to generate colors with probability `Pcolcorr`
    '''

    x    = x0.copy()
    nvec = nvec0.copy()

    stot = 0.
    idc  = 0

    # setting up distributions
    mu = track.width/2

    # initialize cone array
    ncones = len(track.xc1) + len(track.xc2) + len(xfcones)
    xcones = np.vstack( ( np.hstack( (track.xc1, track.xc2, xfcones) ),
                          np.hstack( (track.yc1, track.yc2, yfcones) ) ) )
    xcones = xcones.T
    ycones = np.ones( (ncones,) )

    detCones = np.zeros( (ncones,) , dtype=bool )

    # color of cones
    np.random.seed( seed=seed )
    rightCones = np.hstack( ( np.random.choice( a=[False,True], size=len(track.xc1), p=[Pcolcorr,1-Pcolcorr] ),
                              np.random.choice( a=[True,False], size=len(track.xc2), p=[Pcolcorr,1-Pcolcorr] ),
                              rfcones ) )

    xtraj = np.zeros( (0,2) )

    while ( (( np.linalg.norm( x - x0 ) > track.width) or ( stot < 5. )) and (stot < smax) ):

        # detect cones
        ndetprev = np.sum( detCones )
        camera.checkVisibleMem( xcones, nvec, x, detCones )
        ndetnew = np.sum( detCones )

        # unit vector perpendicular to nvec
        tvec = np.array( [-nvec[1],nvec[0]] )

        # generate paths using different inverse radii
        nns   = np.linspace( -1./rmin, 1./rmin, nsamples )
        fobjs = np.zeros( (nsamples,) )

        for ii in range(0,nsamples):

            ns = nns[ii]

            if abs(ns) < 1e-5:
                r = 15. # random number -- will get fixed further down
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

                rcone = ( abs(r) - np.linalg.norm( xcone - xcenter ) ) * np.sign( ns )

                v1 = xcone - xcenter
                v2 = x - xcenter
                theta = acos( np.dot( v1, v2 ) / ( np.linalg.norm(v1) * np.linalg.norm(v2) ) )

                s = theta * abs(r)

                if abs(ns) < 1e-5:
                    v1    = xcone - x
                    s     = np.dot( v1, nvec )
                    rcone = np.linalg.norm( v1 - np.dot(v1,nvec) * nvec ) * np.sign( np.dot( v1, tvec ) )

                # if cone too far away, skip
                if s > sdetectmax:
                    continue

                sigma = sigmabar * exp(alpha*s)
                prob  = 0.
                if rightCones[icone]:
                    prob += 1/sqrt(2*pi*sigma**2) * exp( - (rcone-mu)**2/(2*sigma**2) ) * Pcolcorr \
                         +  1/sqrt(2*pi*sigma**2) * exp( - (rcone+mu)**2/(2*sigma**2) ) * (1-Pcolcorr)
                else:
                    prob += 1/sqrt(2*pi*sigma**2) * exp( - (rcone-mu)**2/(2*sigma**2) ) * (1-Pcolcorr) \
                         +  1/sqrt(2*pi*sigma**2) * exp( - (rcone+mu)**2/(2*sigma**2) ) * Pcolcorr

                fobjs[ii] += prob

        # find maximum objective value
        imax = np.argmax( fobjs )
        nsmax = nns[imax]

        # update position and heading
        theta = ds*nsmax
        Tmat  = np.array( ( [cos(theta),-sin(theta)],
                            [sin(theta), cos(theta)] ) )

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

    return xtraj, detCones, rightCones
