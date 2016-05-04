from sklearn.tree import DecisionTreeClassifier
import multiclass
import util
import time
from datasets import *


t = multiclass.makeBalancedTree(range(5))
h = multiclass.MCTree(t, lambda: DecisionTreeClassifier(max_depth=3))
h.train(WineDataSmall.X, WineDataSmall.Y)
P = h.predictAll(WineDataSmall.Xte)
print "Accuracy on smaller data set"
print mean(P == WineDataSmall.Yte)

print "------------------------------------"

t = multiclass.makeBalancedTree(range(20))
h = multiclass.MCTree(t, lambda: DecisionTreeClassifier(max_depth=3))
h.train(WineData.X, WineData.Y)
P = h.predictAll(WineData.Xte)
print "Accuracy on full data set"
print mean(P == WineData.Yte)

