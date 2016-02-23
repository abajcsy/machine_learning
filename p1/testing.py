from numpy import *
from pylab import *

import util
import binary
import datasets
import dumbClassifiers
import runClassifier
import knn
import perceptron
import dt
"""
h = dumbClassifiers.AlwaysPredictOne({})
print h

h.train(datasets.TennisData.X, datasets.TennisData.Y)
print h.predictAll(datasets.TennisData.X)

print mean((datasets.TennisData.Y > 0) == (h.predictAll(datasets.TennisData.X) > 0))
print mean((datasets.TennisData.Yte > 0) == (h.predictAll(datasets.TennisData.Xte) > 0))

runClassifier.trainTestSet(h, datasets.TennisData)

h = dumbClassifiers.AlwaysPredictMostFrequent({})
runClassifier.trainTestSet(h, datasets.TennisData)

runClassifier.trainTestSet(dumbClassifiers.AlwaysPredictOne({}), datasets.SentimentData)


h = dumbClassifiers.FirstFeatureClassifier({})
print h.train(datasets.TennisData.X, datasets.TennisData.Y)
print h.predictAll(datasets.TennisData.X)
"""
"""
runClassifier.trainTestSet(dumbClassifiers.AlwaysPredictMostFrequent({}), datasets.SentimentData)
"""
"""
runClassifier.trainTestSet(dumbClassifiers.FirstFeatureClassifier({}), datasets.TennisData)
runClassifier.trainTestSet(dumbClassifiers.FirstFeatureClassifier({}), datasets.SentimentData)
"""
"""
X = datasets.TennisData.X
Y = datasets.TennisData.Y
print X
print Y
print Y[X[:,0] > 0]
print util.mode(Y[X[:,0] > 0])
"""
"""
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 1}), datasets.TennisData)
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 3}), datasets.TennisData)
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 5}), datasets.TennisData)

runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 0.5}), datasets.TennisData)
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 1.0}), datasets.TennisData)
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 2.0}), datasets.TennisData)


runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 1}), datasets.TennisData)
runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 2}), datasets.TennisData)

print '\n'

runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 200}), datasets.TwoDDiagonal)

print '\n'

runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 1}), datasets.SentimentData)
runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 2}), datasets.SentimentData)
"""

runClassifier.trainTestSet(dt.DT({'maxDepth': 5}), datasets.SentimentData)
'''
h = dt.DT({'maxDepth': 5})
h.train(datasets.TennisData.X, datasets.TennisData.Y)
print h
'''













