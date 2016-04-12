from sklearn.tree import DecisionTreeClassifier
import multiclass
import util
import time
from datasets import *

### Part (A) ###

# Sauvignon-Blanc #

sbOAA = multiclass.OAA(5, lambda: DecisionTreeClassifier(max_depth=3))
sbOAA.train(WineDataSmall.X, WineDataSmall.Y)

print "Tree for Sauvignon-Blanc OAA"

util.showTree(sbOAA.f[0], WineDataSmall.words)

sbAVA = multiclass.AVA(5, lambda: DecisionTreeClassifier(max_depth=3))
sbAVA.train(WineDataSmall.X, WineDataSmall.Y)

print"----------------------------------------------"

print "Trees for Sauvignon-Blanc AVA"

print "1 versus 0"
util.showTree(sbAVA.f[1][0], WineDataSmall.words)
print "2 versus 0"
util.showTree(sbAVA.f[2][0], WineDataSmall.words)
print "3 versus 0"
util.showTree(sbAVA.f[3][0], WineDataSmall.words)
print "4 versus 0"
util.showTree(sbAVA.f[4][0], WineDataSmall.words)

print ""
print"----------------------------------------------"

# Pinot-Noir #

print "Tree for Pinot-Noir OAA"

util.showTree(sbOAA.f[2], WineDataSmall.words)

print"----------------------------------------------"
print "Trees for Pinot-Noir AVA"

print "2 versus 0"
util.showTree(sbAVA.f[2][0], WineDataSmall.words)
print "2 versus 1"
util.showTree(sbAVA.f[2][1], WineDataSmall.words)
print "3 versus 2"
util.showTree(sbAVA.f[3][2], WineDataSmall.words)
print "4 versus 2"
util.showTree(sbAVA.f[4][2], WineDataSmall.words)

print ""
print"----------------------------------------------"

### Part (B) ###
print "----------- Part (B) ----------------"

fullOAA = multiclass.OAA(20, lambda: DecisionTreeClassifier(max_depth=3))
tOAA0 = time.time()
fullOAA.train(WineData.X, WineData.Y)
tOAA1 = time.time()

print "OAA training time is %f" % (tOAA1 - tOAA0)

P = fullOAA.predictAll(WineData.Xte)

print "OAA accuracy is %f" % mean(WineData.Yte == P)

print"----------------------------------------------"



fullAVA = multiclass.AVA(20, lambda: DecisionTreeClassifier(max_depth=3))
tAVA0 = time.time()
fullAVA.train(WineData.X, WineData.Y)
tAVA1 = time.time()

print "AVA training time is %f" % (tAVA1 - tAVA0)

P = fullAVA.predictAll(WineData.Xte)

print "AVA accuracy is %f" % mean(WineData.Yte == P)

print"----------------------------------------------"

print "Viognier trees"
util.showTree(fullOAA.f[17], WineData.words)
print"----------------------------------------------"
print ""

for i in range(0,17) :
	print "17 versus %d" % i
	util.showTree(fullAVA.f[17][i], WineData.words)

print ""

### Part (C) ###
print "----------- Part (C) ----------------"
zoP = fullOAA.predictAll(WineData.Xte, useZeroOne=True)

print "OAA Accuracy using zero/one is %f" % mean(WineData.Yte == zoP)

zoP = fullAVA.predictAll(WineData.Xte, useZeroOne=True)

print "AVA Accuracy using zero/one is %f" % mean(WineData.Yte == zoP)





