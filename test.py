import numpy as np
from math import *
import matplotlib.pyplot as plt

# covariance kernel
l     = 1.0
sigma = 1.0

# data points
xd = np.array([-4, -3, -2, -1, 1])
yd = np.sin( xd )

# covariance kernel
#   squared exponential kernel
def kernel(x1,x2,sigma,l):

    ons1 = np.ones( np.size(x2) )
    ons2 = np.ones( np.size(x1) )

    Kxx = sigma**2 * np.exp( - (np.outer(x1,ons1) - np.outer(ons2,x2))**2 / (2*l**2) )

    return Kxx

# "grid"
x = np.linspace(-5.,5,500)

R11 = kernel(x ,x ,sigma,l)
R12 = kernel(x, xd,sigma,l)
R22 = kernel(xd,xd,sigma,l)

y = np.matmul( R12, np.linalg.solve( R22, yd ) )
std_y = R11 - np.matmul( R12, np.linalg.solve(R22,R12.T) )
print(std_y)
ytrue = np.sin( x )

ystd3_1 = y + 1.96 * np.sqrt( np.diag(std_y) )
ystd3_2 = y - 1.96 * np.sqrt( np.diag(std_y) )

plt.fill_between(x,ystd3_1,ystd3_2,alpha=0.3)

plt.plot(x,ytrue,'k:')
plt.plot(xd,yd,'ro')

plt.plot(x,y,'-')

plt.show()
