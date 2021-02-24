import numpy as np
import math

cubePath = "./data/output/Attempt7/"
trainMatrix = np.load(cubePath + "trainMatrix.npy")
testMatrix = np.load(cubePath + "testMatrix.npy")
print ("Train Size: ", math.floor(trainMatrix.shape[1]*.7))
print ("Valid Size: ", math.ceil(trainMatrix.shape[1]*.3))
print ("Test Size: ", testMatrix.shape[1])