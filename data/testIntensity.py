import nrrd
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

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