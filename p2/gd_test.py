import linear
from imports import *
import datasets
import math

x, trajectory = gd.gd(lambda x: x**3+x**10-x**2, lambda x: x*(10*x**8+3*x-2), -1, 100, 0.01)
plot(trajectory)
xlabel('Iteration Number')
ylabel('Approximated Value')
#ylim([0,50])
suptitle('Gradient Descent with Step Size = 0.5')
show(True)

