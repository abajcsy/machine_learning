import linear
from imports import *
import datasets
import math
import pylab
import numpy

#x, trajectory = gd.gd(lambda x: x**3+x**10-x**2, lambda x: 3*(x**2)+10*(x**9)-2*x, -1, 100, 0.01)
#plot(trajectory)
#pylab.plot()
#xlabel('Iteration Number')
#ylabel('Approximated Value')
#ylim([0,50])
#suptitle('Plot of x^3+x^10-x^2')
#show(True)

x = numpy.linspace(-15,15,1000) # 100 linearly spaced numbers
y = x**3+x**10-x**2 # computing the values of sin(x)/x

# compose plot
pylab.plot(x,y) # x**3+x**10-x**2
pylab.plot(0.606,-0.138, 'ro')
pylab.plot(-0.911,-1.192, 'ro')
#pylab.plot(x,y,'co') # same function with cyan dots
#pylab.plot(x,2*y,x,3*y) # 2*sin(x)/x and 3*sin(x)/x
xlabel('X-Value')
ylabel('Y-Value')
ylim([-3,3])
xlim([-3,3])
suptitle('Plot of x^3+x^10-x^2 and Local and Global Minima')
pylab.show() # show the plot
