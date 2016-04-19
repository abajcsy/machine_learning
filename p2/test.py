import linear
from imports import *
import datasets

#a = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 0, 'numIter': 100, 'stepSize': 0.5})
#print runClassifier.trainTestSet(a, datasets.WineDataBinary)
#print a

b = linear.LinearClassifier({'lossFunction': linear.LogisticLoss(), 'lambda': 0, 'numIter': 100, 'stepSize': 0.5})
print runClassifier.trainTestSet(b, datasets.WineDataBinary)
print b

#c = linear.LinearClassifier({'lossFunction': linear.HingeLoss(), 'lambda': 0, 'numIter': #100, 'stepSize': 0.5})
#print runClassifier.trainTestSet(c, datasets.WineDataBinary)
#print c

f = datasets.WineDataBinary()

count = 1
for word in f.words:
	print str(count) + word 
	count += 1


