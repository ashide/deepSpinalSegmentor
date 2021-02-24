import numpy as np
import nrrd
from scipy import ndimage
import volumentations as V
import random

# input pointer
dataPaths = "D:/test/segmentation/database/cervical.txt"
cubePath = "./data/output/Cervical/"
cubeSide = 128
augmentation = 1
testPortion = .2

def normalize_0_1(x):
    x_max = np.max(x)
    x_min = np.min(x)
    x = (x - x_min)
    if((x_max - x_min) == 0):
        return x
    return (x / (x_max - x_min)).clip(0, 1)

def zscore_normalize(image, mask):
    logical_mask = mask == 1  # force the mask to be logical type
    if(len(image[logical_mask]) == 0):
        return image
    mean = image[logical_mask].mean()
    std = image[logical_mask].std()
    normalized = (image - mean) / std
    return normalized

def get_volume_paths(dataPaths: str):
    dataPaths = dataPaths.replace("\\","/")
    folderPath = '/'.join(dataPaths.split("/")[:-1])
    dataPathsFile = open(dataPaths, 'r')
    imagePaths = []
    for item in dataPathsFile.readlines():
        if "," not in item:
            continue
        item = item.replace("\\","/")
        volumePath, maskPath = item.replace("\n", "").split(",")
        imagePaths.append([folderPath + "/" + volumePath, folderPath + "/" + maskPath])
    dataPathsFile.close()
    return imagePaths

def getNrrdShape(nrrd_header):
    data_shape = nrrd_header["sizes"]
    space_magnitudes = np.array([np.linalg.norm(direction) for direction in nrrd_header['space directions']])
    coefs = space_magnitudes/min(space_magnitudes)
    normal_size = data_shape * coefs
    return normal_size

def readAndFixNrrdImage(imagePath):
    print("Reading. ", end="")
    imageArray, imageHeader = nrrd.read(imagePath[0])
    maskArray, _ = nrrd.read(imagePath[1])
    assert(imageArray.shape == maskArray.shape)
    # single class
    maskArray[maskArray > 1] = 1
    maskArray[maskArray < 1] = 0
    arrayShape = np.array(imageArray.shape)
    imageShape = getNrrdShape(imageHeader)
    imageShape = imageShape * cubeSide / imageShape.max()
    print("Scaling. ", end="")
    imageArray = ndimage.zoom(imageArray, imageShape / arrayShape)
    maskArray = ndimage.zoom(maskArray, imageShape / arrayShape)
    return imageArray, maskArray

def divideTestAndTrain(imagePaths):
    random.shuffle(imagePaths)
    cutPoint = int((1 - testPortion) * len(imagePaths))
    return imagePaths[:cutPoint], imagePaths[cutPoint:]

imagePaths = get_volume_paths(dataPaths)
trainImages, testImages = divideTestAndTrain(imagePaths)
totalCount = len(imagePaths)
trainCount = len(trainImages)
trainMatrix = np.zeros((2, trainCount * (1 + augmentation), cubeSide, cubeSide, cubeSide))
testMatrix = np.zeros((2, (totalCount - trainCount) * (1 + augmentation), cubeSide, cubeSide, cubeSide))

def insertDataToMatrix(imageData, maskData, isTrain, index):
    # getting ready
    #imageScale = cubeSide / np.array(imageData.shape).max()
    #print("Resizing. ", end="")
    #imageData = ndimage.zoom(imageData, imageScale)
    #maskData = ndimage.zoom(maskData, imageScale)
    imageShape = np.array(imageData.shape)
    print("Normalizing. ", end="")
    imageData = zscore_normalize(imageData, maskData)
    imageData = normalize_0_1(imageData)       
    print("Reshaping. ", end="") 
    bestOrder = [[imageShape[d], d] for d in range(len(imageShape))]
    bestOrder.sort(reverse=True)
    bestOrder = [d[1] for d in bestOrder]
    imageData = imageData.transpose(bestOrder)
    maskData = maskData.transpose(bestOrder)
    imageShape = np.array(imageData.shape)
    # bugfix
    maskData[maskData >= .8] = 1
    maskData[maskData < .8] = 0
    # appending
    imageShape[imageShape > cubeSide] = cubeSide
    if isTrain:
        trainMatrix[0, index, 0:imageShape[0], 0:imageShape[1], 0:imageShape[2]] = imageData[0:imageShape[0], 0:imageShape[1], 0:imageShape[2]]
        trainMatrix[1, index, 0:imageShape[0], 0:imageShape[1], 0:imageShape[2]] = maskData[0:imageShape[0], 0:imageShape[1], 0:imageShape[2]]
    else:
        testMatrix[0, index, 0:imageShape[0], 0:imageShape[1], 0:imageShape[2]] = imageData[0:imageShape[0], 0:imageShape[1], 0:imageShape[2]]
        testMatrix[1, index, 0:imageShape[0], 0:imageShape[1], 0:imageShape[2]] = maskData[0:imageShape[0], 0:imageShape[1], 0:imageShape[2]]

def applyAugmentation(imageData, maskData):
    augMethod = V.Compose([
        V.RandomScale(),
        V.Rotate((0,360),(0,360),(0,360), p=1),
        V.ElasticTransform(deformation_limits=(0, .5)),
        V.RandomGamma(gamma_limit=(0.5, 1.5)),
        V.GaussianNoise()
    ], p=1)
    aug_data = augMethod(**{
        'image': imageData,
        'mask': maskData,
    })    
    return aug_data['image'], aug_data['mask']

for i in range(totalCount):
    print("File", str(i+1), "out of", str(totalCount), ": ", end="")
    isTrain = i < trainCount
    imageData, maskData = readAndFixNrrdImage(trainImages[i]) if isTrain else readAndFixNrrdImage(testImages[i - trainCount])
    print("Appending. ", end="")
    dataIndex = i if isTrain else (i - trainCount)
    dataIndex = dataIndex * (1 + augmentation)
    insertDataToMatrix(imageData, maskData, isTrain, dataIndex)
    for augIndex in range(augmentation):
        print("Augmentation. ", end="")
        augImageData, augMaskData = applyAugmentation(imageData, maskData)
        insertDataToMatrix(augImageData, augMaskData, isTrain, dataIndex + augIndex + 1)
    print("Done. ")
    
# hdf5
print("Writing npy: ", end="")
np.save(cubePath + 'trainMatrix.npy', trainMatrix.astype("<f4"))
np.save(cubePath + 'testMatrix.npy', testMatrix.astype("<f4"))
print("Done. ")
