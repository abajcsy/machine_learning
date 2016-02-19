from numpy import *
from pylab import *

import matplotlib.pyplot as plt
import util
import binary
import datasets
import runClassifier
import perceptron


def plotCurve(titleString, res):
    plot(res[0], res[1], 'b-',
         res[0], res[2], 'r-')
    legend( ('Train', 'Test') )
    xlabel('# of training points')
    ylabel('Accuracy')
    title(titleString)
    show() 
    
# A - learning curve with 5 epochs    

(dataSizes, trainAcc, testAcc) = runClassifier.learningCurveSet(perceptron.Perceptron({'numEpoch': 5}), datasets.SentimentData)
plotCurve("Learning Curve for Perceptron", [dataSizes, trainAcc, testAcc])
savefig("LC_perceptron.png")



# B - number of epochs vs train/test accuracy

# Removed third return value
def trainTest(classifier, X, Y, Xtest, Ytest):
	"""
	Train a classifier on data (X,Y) and evaluate on
	data (Xtest,Ytest).  Return a triple of:
	  * Training data accuracy
	  * Test data accuracy
	  * Individual predictions on Xtest.
	"""

	classifier.reset()                           # initialize the classifier
	classifier.train(X, Y);                      # train it

	#print "Learned Classifier:"
	#print classifier

	Ypred = classifier.predictAll(X);               # predict the training data
	trAcc = mean((Y     >= 0) == (Ypred >= 0));     # check to see how often the predictions are right

	Ypred = classifier.predictAll(Xtest);           # predict the training data
	teAcc = mean((Ytest >= 0) == (Ypred >= 0));     # check to see how often the predictions are right

	print "Training accuracy %g, test accuracy %g" % (trAcc, teAcc)
	return (trAcc, teAcc)
    
 # Had to add return statement here
def trainTestSet(classifier, dataset):
	return trainTest(classifier, dataset.X, dataset.Y, dataset.Xte, dataset.Yte)  
  
# Where something actually happens
epochs = arange(1,11)
trainAcc = []
testAcc = []

for i in epochs :
	h = perceptron.Perceptron({'numEpoch': i})
	(trAcc, teAcc)  = trainTestSet(h, datasets.SentimentData)
	trainAcc.append([trAcc])
	testAcc.append([teAcc])
	
plotCurve("Effect of number of epochs on train/test accuracy", [epochs, trainAcc, testAcc])
savefig("Epochs_train_test.png") 
	
