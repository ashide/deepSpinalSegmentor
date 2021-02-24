import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from skimage.segmentation import mark_boundaries
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from skimage.exposure import rescale_intensity
from keras.callbacks import History
from keras.utils.vis_utils import plot_model
from skimage import io
import tensorflow as tf
import pickle
from model.metrics import dice_coef, dice_coef_loss, generalized_dice_coeff
from model.train import get_unet_2d, get_unet_3d, prepareFor2D, prepareFor3D
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries

modelPath = "./model/output/Attempt2/"
testCubesPath = "./data/output/"
pictureOutputPath = './model/output/Attempt2/pics/'
testList = ['Osf', 'Siegen', 'Wholespine', 'Zenodo', 'Cervical']

testMatrix = np.load(testCubesPath + testList[0] + "/testMatrix.npy")
cubeSide = testMatrix.shape[-1]
unet_2d = get_unet_2d(cubeSide)
unet_3d = get_unet_3d(cubeSide)
unet_2d.load_weights(modelPath + "Unet_2D.h5")
unet_3d.load_weights(modelPath + "Unet_3D.h5")
for testCase in testList:
    testMatrix = np.load(testCubesPath + testCase + "/testMatrix.npy")
    testImgs_2d, testLabels_2d = prepareFor2D(testMatrix, shuffle=False)
    testImgs_3d, testLabels_3d = prepareFor3D(testMatrix, shuffle=False)
    testPredict_2d = unet_2d.predict(testImgs_2d, verbose=0)
    testPredict_3d = unet_3d.predict(testImgs_3d, verbose=0, batch_size=1)
    testDSC_2d = dice_coef(testPredict_2d, testLabels_2d).numpy()
    testDSC_3d = dice_coef(testPredict_3d, testLabels_3d).numpy()
    pics_image=testImgs_2d.reshape([-1,cubeSide,cubeSide,cubeSide])
    pics_labels=testLabels_2d.reshape([-1,cubeSide,cubeSide,cubeSide])
    pics_predict_2d=testPredict_2d.reshape([-1,cubeSide,cubeSide,cubeSide])
    pics_predict_3d=testPredict_3d.reshape([-1,cubeSide,cubeSide,cubeSide])
    shape=pics_image.shape
    for k in range(shape[0]):
        for i in range(cubeSide):
            a = (pics_image[k][:,:,i]*200).astype("uint8")
            b = (pics_labels[k][:,:,i]*255).astype("uint8")
            c = (pics_predict_2d[k][:,:,i]*55).astype("uint8")
            d = (pics_predict_3d[k][:,:,i]*55).astype("uint8")
            if b.any():
                final=np.zeros((cubeSide,cubeSide*2,3))
                temp = np.stack((a,)*3, axis=-1)
                temp[:,:,1] = a + c
                temp = mark_boundaries(temp,b)
                final[:,0:cubeSide,:]=temp
                temp = np.stack((a,)*3, axis=-1)
                temp[:,:,1] = a + d
                temp = mark_boundaries(temp,b)
                final[:,cubeSide:,:]=temp
                final=(final*255).astype("uint8")
                io.imsave(os.path.join(pictureOutputPath, testCase + "_" + str(k) + '_' + str(i) + '.png'), final)
    
    print(testCase)
    print('\t Unet 2D: ', testDSC_2d)
    print('\t Unet 3D: ', testDSC_3d)


