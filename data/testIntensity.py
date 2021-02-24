import nrrd
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras import backend as K

smooth = 1
realLabels = pickle.load(open("real.pickle","rb"))
predLabels = pickle.load(open("predicts.pickle","rb"))

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef3D(y_true, y_pred):
    shape = y_true.shape
    new_shape = [-1]
    y_true_s = K.reshape(y_true, [128, -1])
    y_pred_s = K.reshape(y_pred, new_shape)
    intersection = K.sum(y_true_s * y_pred_s, 1)
    return K.sum((2. * intersection + smooth) / (K.sum(y_true_s, 1) + K.sum(y_pred_s, 1) + smooth)) / shape[0]

count2D=0
count3D=0
value2D=0
value3D=0
for i in range(realLabels.shape[0]):
    count3D=count3D+1
    real3D=realLabels[i,:]
    pred3D=predLabels[i,:]
    value3D=value3D+dice_coef3D(real3D, pred3D)
    for j in range(realLabels.shape[1]):
        count2D=count2D+1
        real2D=realLabels[i,j,:]
        pred2D=predLabels[i,j,:]
        value2D=value2D+dice_coef(real2D, pred2D)
value3D=value3D/count3D
value2D=value2D/count2D

print(value3D)
print(value2D)



imagePath = "Desktop\\AKA2.nrrd"
maskPath = "Desktop\\AKA2_Labels.nrrd"
outputPath = "Desktop\\AKA2_NewLabels.nrrd"

imageArray, imageHeader = nrrd.read(imagePath)
maskArray, maskHeader = nrrd.read(maskPath)
newImage=ndimage.convolve(imageArray, np.ones((3,3,3))/27)
onlyMasked = newImage[maskArray>0]
std=np.std(onlyMasked)
mean=np.mean(onlyMasked)
coef=.2
low=mean-std*coef
high=mean+std*coef
newImage[newImage<low]=0
newImage[newImage>high]=0
plt.hist(onlyMasked, bins=255)
plt.show()
nrrd.write(outputPath,newImage,imageHeader)
a=1