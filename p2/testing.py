from numpy import *
from pylab import *

import util
import imports
import datasets
import gd

gd.gd(lambda x: x**2, lambda x: 2*x, 10, 10, 0.2)