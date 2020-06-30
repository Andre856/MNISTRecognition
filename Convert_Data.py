import numpy as np

from keras.datasets import mnist

def LoadSmallData():

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Half images to 14*14
    train_images = train_images[:,::2,::2]
    test_images = test_images[:,::2,::2]

    train_images_2D, test_images_2D = Convert2D(train_images, test_images, 14)

    train_images_2D, test_images_2D = NormaliseData(train_images_2D, test_images_2D)

    train_labels = OneHotEncode(train_labels)
    test_labels = OneHotEncode(test_labels)

    return (train_images_2D, train_labels), (test_images_2D, test_labels), train_images

def LoadNormalData():

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images_2D, test_images_2D = Convert2D(train_images, test_images, 28)

    train_images_2D, test_images_2D = NormaliseData(train_images_2D, test_images_2D)

    train_labels = OneHotEncode(train_labels)
    test_labels = OneHotEncode(test_labels)

    return (train_images_2D, train_labels), (test_images_2D, test_labels), train_images

def NormaliseData(trainData, testData):
    return trainData / 255, testData / 255

def Convert2D(trainData, testData, num):

    train_images_2D = np.zeros((trainData.shape[0], trainData.shape[1]*trainData.shape[2]))
    for k in range(trainData.shape[0]):
        for i in range(trainData.shape[1]):
            train_images_2D[k, i*num:(i+1)*num] += trainData[k, :, i]

    test_images_2D = np.zeros((testData.shape[0], testData.shape[1]*testData.shape[2]))
    for k in range(testData.shape[0]):
        for i in range(testData.shape[1]):
            test_images_2D[k, i*num:(i+1)*num] += testData[k, :, i]

    return train_images_2D, test_images_2D

def OneHotEncode(labels):
    K = np.max(labels) + 1
    N = len(labels)
    ind = np.zeros((N,K))
    for i in range(N):
        ind[i,labels[i]] = 1
    return ind

