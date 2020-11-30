import h5py
import numpy as np
from PIL import Image as pImage
from model.train import prepareFor2D

cubePath = "./data/output/Attempt3/"
outputFolder = "./data/output/Attempt3/"
trainMatrix = np.load(cubePath + "trainMatrix.npy")
# testMatrix = np.load(cubePath + "testMatrix.npy")

dataMatrix, truthMatrix = prepareFor2D(trainMatrix)

assert dataMatrix.shape == truthMatrix.shape
assert len(dataMatrix.shape) == 4
assert dataMatrix.max() <= 1
assert dataMatrix.min() >= 0
assert truthMatrix.max() in [0, 1]
assert truthMatrix.min() in [0, 1]

imageShape = dataMatrix.shape[0:3]
order = [[imageShape[d], d] for d in range(len(imageShape))]
order.sort(reverse=True)
order = [d[1] for d in order] + [3]
dataMatrix = dataMatrix.transpose(order)
truthMatrix = truthMatrix.transpose(order)
imageShape = dataMatrix.shape
for i in range(imageShape[0]):
    dataPicture = (dataMatrix[i, :, :, 0]*200).astype("uint8")
    truthPicture = (truthMatrix[i, :, :, 0]*50).astype("uint8")
    finalPicture = np.stack((dataPicture,)*3, axis=-1)
    finalPicture[:,:,1] = dataPicture + truthPicture
    pImage.fromarray(finalPicture).save(outputFolder + "check_"+str(i)+".png")
