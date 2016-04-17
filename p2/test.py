import linear
from imports import *
f = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 0, 'numIter': 100, 'stepSize': 0.5})
print runClassifier.trainTestSet(f, datasets.TwoDAxisAligned)

print f

mlGraphics.plotLinearClassifier(f, datasets.TwoDAxisAligned.X, datasets.TwoDAxisAligned.Y)

show(False)

f = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 10, 'numIter': 100, 'stepSize': 0.5})
print runClassifier.trainTestSet(f, datasets.TwoDAxisAligned)
print f


f = linear.LinearClassifier({'lossFunction': linear.LogisticLoss(), 'lambda': 10, 'numIter': 100, 'stepSize': 0.5})
print runClassifier.trainTestSet(f, datasets.TwoDDiagonal)
print f

f = linear.LinearClassifier({'lossFunction': linear.HingeLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
print runClassifier.trainTestSet(f, datasets.TwoDDiagonal)
print f