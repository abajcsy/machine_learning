import linear
from imports import *
import datasets

a = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 1, 'numIter': 8000, 'stepSize': 0.0125})
print "sqLoss "
runClassifier.trainTestSet(a, datasets.WineDataBinary)
#print a


b = linear.LinearClassifier({'lossFunction': linear.LogisticLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
print "logloss "
runClassifier.trainTestSet(b, datasets.WineDataBinary)
#print b

w = b.weights
si = argsort(w)


f = datasets.WineDataBinary()
# Worst not indicative of class
print f.words[si[0]]
print f.words[si[1]]
print f.words[si[2]]

print "-------------------------------"
# Words indicative of class
print f.words[si[-1]]
print f.words[si[-2]]
print f.words[si[-3]]



c = linear.LinearClassifier({'lossFunction': linear.HingeLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
print "hingeloss "
runClassifier.trainTestSet(c, datasets.WineDataBinary)
#print c

'''
f = datasets.WineDataBinary()

count = 1
for word in f.words:
	print str(count) + word 
	count += 1
'''


