from numpy import *
from pylab import *

import util
import binary
import datasets
import knn
import runClassifier

# Learning curves for K = 1, 2, 10, 20

for i in [1, 2, 10, 20] :
	(dataSizes, trainAcc, testAcc) = runClassifier.learningCurveSet(knn.KNN({'isKNN': True, 'K': i}), datasets.DigitData)
	runClassifier.plotCurve("Learning Curve for knn, K=%d" % (i), [dataSizes, trainAcc, testAcc])
	ylim([.2,1.1])
	savefig("LC_k%d.png" % (i) )
	close()

# Learning curves for \epsilon = 5.0, 10.0, 15.0, 20.0
for i in [5.0, 10.0, 15.0, 20.0] :
	(dataSizes, trainAcc, testAcc) = runClassifier.learningCurveSet(knn.KNN({'isKNN': False, 'eps': i}), datasets.DigitData)
	runClassifier.plotCurve("Learning Curve for knn, eps=%f" % (i), [dataSizes, trainAcc, testAcc])
	ylim([.45,1.1])
	savefig("LC_eps%f.png" % (i) )
	close()

# Learning curve for K = 5
(dataSizes, trainAcc, testAcc) = runClassifier.learningCurveSet(knn.KNN({'isKNN': True, 'K': 5}), datasets.DigitData)

runClassifier.plotCurve("Learning Curve for knn, K=5", [dataSizes, trainAcc, testAcc])
savefig("LC_k5.png")
close()
