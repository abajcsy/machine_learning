from numpy import *
from pylab import *

import matplotlib.pyplot as plt
import util
import binary
import datasets
import knn
import runClassifier
import KNNDigits

# A - Histogram for all 784 dimensions

def computeDistances(data):
	N = len(data)
	dist = []
	for n in range(N):
		for m in range(n):
			dist.append(linalg.norm(data[n]-data[m]))
	return dist

data = datasets.DigitData

plt.xlabel('distance')
plt.ylabel('# of pairs of points at that distance')
plt.title('Histogram of pairwise distance in 784 dimensions')

distances = computeDistances(datasets.DigitData.X)

plt.hist(distances)

plt.savefig('AllDims.png')
plt.close()

# B/C Subsampled to fixed dimensionality

def computeSampledDistances(data, dims):
	N = len(data)
	d = len(dims)
	dist = []
	for n in range(N):
		for m in range(n):
			dist.append(linalg.norm(data[n,dims]-data[m,dims])/sqrt(d))
	return dist

	
D = [2, 8, 32, 128, 512]
Cols = ['#FF0000', '#880000', '#000000', '#000088', '#0000FF']
Bins = arange(0, 1, 0.02)

plt.xlabel('distance / sqrt(dimensionality)')
plt.ylabel('# of pairs of points at that distance')
plt.title('dimensionality versus uniform point distances')

for i,d in enumerate(D) :
	ind = arange(0,784)
	util.permute(ind)
	distances = computeSampledDistances(datasets.DigitData.X, ind[arange(0,d)])
	plt.hist(distances, Bins, histtype='step', color=Cols[i])

plt.legend(['%d dims' % d for d in D])
plt.savefig('Subsampled.png')
plt.show()
