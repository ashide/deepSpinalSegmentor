import numpy as np
from PIL import Image as pImage
import math

cubePath = "./data/output/Attempt6-Aug/"
outputFolder = "./data/output/Attempt6-Aug/"
trainMatrix = np.load(cubePath + "trainMatrix.npy")
testMatrix = np.load(cubePath + "testMatrix.npy")

assert len(trainMatrix.shape) == 5
assert len(testMatrix.shape) == 5
assert trainMatrix.shape[2] == trainMatrix.shape[3] == trainMatrix.shape[4] == testMatrix.shape[2] == testMatrix.shape[3] == testMatrix.shape[4]
assert trainMatrix.max() == 1
assert testMatrix.max() == 1
assert trainMatrix.min() == 0
assert testMatrix.min() == 0

trainCount = trainMatrix.shape[1]
imageCount = testMatrix.shape[1] + trainCount
cubeSide = trainMatrix.shape[2]
resultSide = math.ceil(math.sqrt(imageCount))
result = np.zeros((cubeSide*resultSide, cubeSide*resultSide, 3)).astype("uint8")
for i in range(cubeSide):
    imageIndex = -1
    for j in range(resultSide):
        for k in range(resultSide):
            imageIndex = imageIndex + 1
            if imageIndex >= imageCount:
                break
            isTrain = imageIndex < trainCount 
            dataPicture = trainMatrix[0, imageIndex, i, :, :] if isTrain else testMatrix[0, imageIndex - trainCount, i, :, :]
            truthPicture = trainMatrix[1, imageIndex, i, :, :] if isTrain else testMatrix[1, imageIndex - trainCount, i, :, :]
            dataPicture = dataPicture * 200
            truthPicture = truthPicture * 50
            finalPicture = np.stack((dataPicture,)*3, axis=-1)
            if isTrain:
                finalPicture[:,:,1] = dataPicture + truthPicture
            else:
                finalPicture[:,:,0] = dataPicture + truthPicture
            # inserting
            result[j*cubeSide:(j+1)*cubeSide, k*cubeSide:(k+1)*cubeSide, :] = finalPicture
    pImage.fromarray(result).save(outputFolder + "check_"+str(i)+".png")
