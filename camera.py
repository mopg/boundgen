from math import *
import numpy as np

class FOV:

    def __init__( self, rad = 5., distcent = 8. ):

        self.rad      = rad
        self.distcent = distcent
        self.cone     = self.distcent > self.rad
        self.beta     = pi
        if self.cone:
            self.beta = asin( rad / distcent )


    def checkVisible( self, xpts, nvec, xpos ):

        npts = np.shape( xpts )[0]

        vis = np.zeros( (npts,) , dtype=bool )

        self.checkVisibleMem( xpts, nvec, xpos, vis )

        return vis


    def checkVisibleMem( self, xpts, nvec, xpos, vis ):

        npts = np.shape( xpts )[0]

        tvec = np.array( [ nvec[1], nvec[0] ] )
        l = 0.; x1 = np.array( [0.,0.] ); x2 = np.array( [0.,0.] )

        if self.cone:
            l   = sqrt( self.distcent**2 - self.rad**2 )
            x1 += xpos + nvec * l * cos(self.beta) + tvec * l * sin(self.beta)
            x2 += xpos + nvec * l * cos(self.beta) - tvec * l * sin(self.beta)

        for jj in range(0,npts):

            xcent = xpos + nvec * self.distcent

            xcurr = xpts[jj,:]

            # first check if within ball
            if np.linalg.norm( xcurr - xcent ) <= self.rad:
                vis[jj] = True
                continue

            # then check if between ball and car (the cone if you will)
            #   compute area of triangle comprised of (x1,x2,xpos)
            if not self.cone:
                continue
            v1 = x1 - xpos; v2 = x2 - xpos
            Atriang = abs( 0.5 * (  v1[0]*v2[1] - v1[1]*v2[0] ) )

            #   compute area of subtriangles
            v1 = xpos - xcurr; v2 = x2 - xcurr
            A1 = abs( 0.5 *(  v1[0]*v2[1] - v1[1]*v2[0] ) )

            v1 = x1 - xcurr; v2 = x2 - xcurr
            A2 = abs( 0.5 *(  v1[0]*v2[1] - v1[1]*v2[0] ) )

            v1 = xpos - xcurr; v2 = x1 - xcurr
            A3 = abs( 0.5 *(  v1[0]*v2[1] - v1[1]*v2[0] ) )

            #   check if point is within triangle
            if (A1 + A2 + A3) < (Atriang+1e-4):
                vis[jj] = True

        return vis

    def getOutline( self, nvec, xpos ):

        tvec = np.array( [ nvec[1], nvec[0] ] )

        nplot = 50

        alpha = pi/2 - self.beta
        phi = np.linspace( -pi/2 + alpha, 3*pi/2-alpha, nplot )

        xpl = np.zeros( (nplot+2,) )
        ypl = np.zeros( (nplot+2,) )

        for jj in range(1,nplot+1):
            xloc = self.rad * cos( phi[jj-1] )
            yloc = self.rad * sin( phi[jj-1] ) + self.distcent

            xpl[jj] =  xloc * tvec[0] + yloc * nvec[0]
            ypl[jj] = -xloc * tvec[1] + yloc * nvec[1]


        xpl += xpos[0]
        ypl += xpos[1]
        
        if not self.cone:
            xpl[0]  = xpl[1]
            xpl[-1] = xpl[-2]

        return (xpl,ypl)
