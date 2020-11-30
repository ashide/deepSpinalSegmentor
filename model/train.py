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

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
cubePath = "./data/output/Attempt6/"
modelPath = "./model/output/Attempt6/3D/"

print('-'*30, '\nLoading data...\n', '-'*30)
trainMatrix = np.load(cubePath + "trainMatrix.npy")
testMatrix = np.load(cubePath + "testMatrix.npy")
cubeSide = trainMatrix.shape[-1]
smooth = 1

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
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

def get_unet_2d():
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

    model.compile(optimizer=Adam(lr=1e-3), loss="binary_crossentropy", metrics=[dice_coef, "binary_accuracy", "mse", "mae", "acc"])

    model.summary()
    plot_model(model, to_file=modelPath + "Model_2D.png", show_shapes=True)
    return model    


def get_unet_3d():
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

    model.compile(optimizer=Adam(lr=1e-3), loss="binary_crossentropy", metrics=[dice_coef, "binary_accuracy", "mse", "mae", "acc"])

    model.summary()
    plot_model(model, to_file=modelPath + "Model_3D.png", show_shapes=True)
    return model    
        

import matplotlib.pyplot as plt

def prepareFor2D(inputMatrix):    
    transposed = np.transpose(inputMatrix, (1,2,3,4,0))
    combined = np.concatenate(transposed, axis=0)
    # shuffle
    np.random.shuffle(combined)
    combined = np.transpose(combined, (3,0,1,2))[..., np.newaxis]
    return combined[0], combined[1]

def prepareFor3D(inputMatrix):
    transposed = np.transpose(inputMatrix, (1,2,3,4,0))
    # shuffle
    np.random.shuffle(transposed)
    combined = np.transpose(transposed, (4,0,1,2,3))[..., np.newaxis]
    return combined[0], combined[1]

def train_and_predict2D():       
    print('-'*30, '\nCreating and compiling model...\n', '-'*30)
    model = get_unet_2d()
    model_checkpoint = ModelCheckpoint(modelPath + "Model_2D.h5", monitor='val_loss', save_best_only=True)
    
    print('-'*30, '\nTraining...\n', '-'*30)
    trainImgs, trainLabels = prepareFor2D(trainMatrix)
    history = model.fit(trainImgs, trainLabels, batch_size=100, epochs=200, verbose=1, shuffle=True, validation_split=0.2, callbacks=[model_checkpoint])

    #pickle.dump(history, open(modelPath + "history.pickle", "wb" ) )
    print('-'*30, '\Predicting Tests...\n', '-'*30)
    testImgs, testLabels = prepareFor2D(testMatrix)
    model.load_weights(modelPath + "Model_2D.h5")
    imgs_pred_test = model.predict(testImgs, verbose=1)
    
    print('-'*30, '\nSaving Predictions...\n', '-'*30)
    for k in range(cubeSide):
        a = (testImgs[k][:,:,0]*200).astype("uint8")
        b = (testLabels[k][:,:,0]*255).astype("uint8")
        c = (imgs_pred_test[k][:,:,0]*55).astype("uint8")
        a3 = np.stack((a,)*3, axis=-1)
        a3[:,:,1] = a + c
        io.imsave(os.path.join(modelPath, "2d_pred_" + str(k) + '.png'), mark_boundaries(a3,b))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def train_and_predict3D():
    print('-'*30, '\nCreating and compiling model...\n', '-'*30)
    model = get_unet_3d()
    model_checkpoint = ModelCheckpoint(modelPath + "Model_3D.h5", monitor='val_loss', save_best_only=True)
    
    print('-'*30, '\nTraining...\n', '-'*30)
    #model.load_weights(modelPath + "Model_3D.h5")
    trainImgs, trainLabels = prepareFor3D(trainMatrix)
    history = model.fit(trainImgs, trainLabels, batch_size=2, epochs=500, verbose=1, shuffle=True, validation_split=0.3, callbacks=[model_checkpoint])

    #pickle.dump(history, open(modelPath + "history.pickle", "wb" ) )
    print('-'*30, '\Predicting Tests...\n', '-'*30)
    testImgs, testLabels = prepareFor3D(testMatrix)
    model.load_weights(modelPath + "Model_3D.h5")
    imgs_pred_test = model.predict(testImgs, batch_size=1, verbose=1)
    
    print('-'*30, '\nSaving Predictions...\n', '-'*30)
    for i in range(testImgs.shape[0]):
        for k in range(cubeSide):
            a = (testImgs[i][k][:,:,0]*200).astype("uint8")
            b = (testLabels[i][k][:,:,0]*255).astype("uint8")
            c = (imgs_pred_test[i][k][:,:,0]*55).astype("uint8")
            a3 = np.stack((a,)*3, axis=-1)
            a3[:,:,1] = a + c
            io.imsave(os.path.join(modelPath, "3d_pred_" + str(i) + '_' + str(k) + '.png'), mark_boundaries(a3,b))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    train_and_predict3D()
