import numpy as np

def genFalsePositives( track, percFP = 0.3, pright = 0.5, sigmadist = 4.0, seed=None ):
    '''
        Generates false positives close to the midline of the track.
            - track:     track with cones
            - percFP:    percentage of false positives (based on number cones in track)
            - pright:    probability of a false cone being detected as a right cone (through color)
            - sigmadist: standard deviation of false cone distance from midline
    '''

    np.random.seed( seed=seed )

    nfps = int( np.ceil( (len(track.xc1)+len(track.xc2)) * percFP ) )

    scones = np.random.uniform( low=0., high=track.length, size=nfps )
    ncones = np.random.normal( 0., sigmadist, size=nfps )

    # interpolate to get local th, x, and y
    xcenter_cone = np.interp( scones, track.sm, track.xm )
    ycenter_cone = np.interp( scones, track.sm, track.ym )
    th_cone      = np.interp( scones, track.sm, track.th )

    xcones    = xcenter_cone + ncones * np.sin( th_cone )
    ycones    = ycenter_cone - ncones * np.cos( th_cone )
    rightcone = np.random.choice( a=[True,False], size=nfps, p=[pright,1-pright] )

    return xcones, ycones, rightcone
