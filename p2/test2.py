import linear
from imports import *
import datasets


f = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 0, 'numIter': 100, 'stepSize': 0.5})
print runClassifier.trainTestSet(f, datasets.TwoDAxisAligned)
print f

a = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 10, 'numIter': 100, 'stepSize': 0.5})
print runClassifier.trainTestSet(a, datasets.TwoDAxisAligned)
print a

b = linear.LinearClassifier({'lossFunction': linear.LogisticLoss(), 'lambda': 10, 'numIter': 100, 'stepSize': 0.5})
print runClassifier.trainTestSet(b, datasets.TwoDDiagonal)
print b

c = linear.LinearClassifier({'lossFunction': linear.HingeLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
print runClassifier.trainTestSet(c, datasets.TwoDDiagonal)
print c

