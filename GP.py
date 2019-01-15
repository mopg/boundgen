from math import *
import numpy as np

class GP( object ):

    def __init__( self, sigma = 1., l = 1., xtrain = np.zeros([1,2]), ytrain = np.zeros(1) ):

        self.sigma  = sigma
        self.l      = l
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.Ktrain = kernel( self.xtrain, self.xtrain, self.sigma, self.l )
        self.Ktytprod = np.linalg.solve( self.Ktrain, self.ytrain )

    def evalPosterior( self, x ):

        # NOTE: assuming zero mean gaussian now (need to update for different prior)

        # compute covariances
        K11 = kernel(x,          x , self.sigma, self.l)
        K12 = kernel(x, self.xtrain, self.sigma, self.l)

        # compute evalPosterior
        y = np.matmul( K12, self.Ktytprod )
        std_y = K11 - np.matmul( K12, np.linalg.solve( self.Ktrain, K12.T ) )

        return (y, std_y)

    def evalPosteriorMean( self, x ):

        # NOTE: assuming zero mean gaussian now (need to update for different prior)

        # compute covariances
        K11 = kernel(x,          x , self.sigma, self.l)
        K12 = kernel(x, self.xtrain, self.sigma, self.l)

        # compute evalPosterior
        y = np.matmul( K12, self.Ktytprod )

        return y

def kernel(x1,x2,sigma,l):

    ons1 = np.ones( np.shape(x2)[0] )
    ons2 = np.ones( np.shape(x1)[0] )

    Kxx = sigma**2 * np.exp( - 1./(2*l**2) * ( ( (np.outer(x1[:,0],ons1) - np.outer(ons2,x2[:,0]))**2 )**2 + (np.outer(x1[:,1],ons1) - np.outer(ons2,x2[:,1]) )**2 ) )

    return Kxx
