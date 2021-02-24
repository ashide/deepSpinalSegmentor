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
import matplotlib.pyplot as plt
from datetime import datetime
from model.metrics import dice_coef, dice_coef_loss, generalized_dice_coeff

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def getEvaluationMetrics():
    return [dice_coef, "mse", "mae", "acc", generalized_dice_coeff]

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def conv3d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv3D(filters=n_filters, kernel_size=(kernel_size, kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv3D(filters=n_filters, kernel_size=(kernel_size, kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def get_unet_2d(cubeSide, modelPath=None):
    n_filters=16 
    dropout=0.5
    batchnorm=True
    input_img = Input((cubeSide, cubeSide, 1), name='img')
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])

    model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=getEvaluationMetrics())

    model.summary()
    if modelPath:
        plot_model(model, to_file=modelPath + "Unet_2D.png", show_shapes=True)
    return model    

def get_unet_3d(cubeSide, modelPath=None):
    n_filters=16
    dropout=0.5
    batchnorm=True
    input_img = Input((cubeSide, cubeSide, cubeSide, 1), name='img')
    # contracting path
    c1 = conv3d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling3D((2, 2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv3d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling3D((2, 2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv3d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling3D((2, 2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv3d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling3D(pool_size=(2, 2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv3d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv3DTranspose(n_filters*8, (3, 3, 3), strides=(2, 2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv3d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv3DTranspose(n_filters*4, (3, 3, 3), strides=(2, 2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv3d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv3DTranspose(n_filters*2, (3, 3, 3), strides=(2, 2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv3d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv3DTranspose(n_filters*1, (3, 3, 3), strides=(2, 2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=4)
    u9 = Dropout(dropout)(u9)
    c9 = conv3d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])

    model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=getEvaluationMetrics())

    model.summary()
    if modelPath:
        plot_model(model, to_file=modelPath + "Unet_3D.png", show_shapes=True)
    return model    
        
def prepareFor2D(inputMatrix, shuffle=True):    
    transposed = np.transpose(inputMatrix, (1,2,3,4,0))
    combined = np.concatenate(transposed, axis=0)
    # shuffle
    if shuffle:
        np.random.shuffle(combined)
    combined = np.transpose(combined, (3,0,1,2))[..., np.newaxis]
    return combined[0], combined[1]

def prepareFor3D(inputMatrix, shuffle=True):
    transposed = np.transpose(inputMatrix, (1,2,3,4,0))
    # shuffle
    if shuffle:
        np.random.shuffle(transposed)
    combined = np.transpose(transposed, (4,0,1,2,3))[..., np.newaxis]
    return combined[0], combined[1]

def train_and_predict(cubePath, modelPath):  
    print('-'*30, '\nLoading data...\n', '-'*30)
    trainMatrix = np.load(cubePath + "trainMatrix.npy")
    testMatrix = np.load(cubePath + "testMatrix.npy")
    cubeSide = trainMatrix.shape[-1]     
    print('-'*30, '\nCreating and compiling model...\n', '-'*30)
    unet_2d = get_unet_2d(cubeSide, modelPath)
    unet_3d = get_unet_3d(cubeSide, modelPath)
    unet_2d_checkpoint = ModelCheckpoint(modelPath + "Unet_2D.h5", monitor='val_loss', save_best_only=True)
    unet_3d_checkpoint = ModelCheckpoint(modelPath + "Unet_3D.h5", monitor='val_loss', save_best_only=True)
    epoch=0
    trainImgs_2d, trainLabels_2d = prepareFor2D(trainMatrix)
    testImgs_2d, testLabels_2d = prepareFor2D(testMatrix)
    trainImgs_3d, trainLabels_3d = prepareFor3D(trainMatrix)
    testImgs_3d, testLabels_3d = prepareFor3D(testMatrix)
    historyObject=[]
    while epoch<300:
        print('\nEpochs from '+str(epoch)+'\n')
        epoch = epoch + 10
        print('\nTraining 2D...10 Epochs\n')
        dateTimeObj = datetime.now()
        history_2d = unet_2d.fit(trainImgs_2d, trainLabels_2d, batch_size=100, epochs=10, verbose=1, shuffle=True, validation_split=0.3, callbacks=[unet_2d_checkpoint])
        unet_2d_time = datetime.now() - dateTimeObj
        dateTimeObj = datetime.now()
        print('\nTraining 3D...10 Epochs\n')
        history_3d = unet_3d.fit(trainImgs_3d, trainLabels_3d, batch_size=2, epochs=10, verbose=1, shuffle=True, validation_split=0.3, callbacks=[unet_3d_checkpoint])
        unet_3d_time = datetime.now() - dateTimeObj
        # 10 epochs have passed
        print('\nAnalysis...\n')
        testPredict_2d = unet_2d.predict(testImgs_2d, verbose=0)
        testPredict_3d = unet_3d.predict(testImgs_3d, verbose=0, batch_size=1)
        testDSC_2d = dice_coef(testPredict_2d, testLabels_2d).numpy()
        testDSC_3d = dice_coef(testPredict_3d, testLabels_3d).numpy()
        historyStep=dict()
        historyStep["history_2d"] = history_2d.history
        historyStep["history_3d"] = history_3d.history
        historyStep["testDSC_2d"] = testDSC_2d
        historyStep["testDSC_3d"] = testDSC_3d
        historyStep["unet_2d_time"] = unet_2d_time
        historyStep["unet_3d_time"] = unet_3d_time
        historyObject.append(historyStep)
        historyFile = open(modelPath + "historyFile.pickle", "wb")
        pickle.dump(historyObject, historyFile)
        historyFile.close()
        
if __name__ == '__main__':
    for i in range(1,8):
        print('\nAttempt '+str(i)+'...\n')
        cubePath = "./data/output/Attempt"+str(i)+"/"
        modelPath = "./model/output/Attempt"+str(i)+"/"
        train_and_predict(cubePath, modelPath)
    
