import numpy as np
import matplotlib.pyplot as plt

# from Convert_Data import LoadSmallData
from Convert_Data import LoadNormalData

# Import Data
# (train_images, train_labels), (test_images, test_labels), image = LoadSmallData()
(train_images, train_labels), (test_images, test_labels), image = LoadNormalData()

Xtrain = train_images
Ytrain = train_labels

Xtest = test_images
Ytest = train_labels

def feedForward(a):
    return sigmoid(a)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def cross_entropy(T, Py):
    tot = T * np.log(Py)
    return tot.sum()

def classification_rate(T, Y):
    return np.mean(np.argmax(T, axis=1) == np.argmax(Y, axis=1))

learning_rate = 0.000025

# Initialise wights and biases for 2 hidden layers and K neurons in each layer
K = 25
W1 = np.random.randn(Xtrain.shape[1], K)
b1 = np.random.randn(K)
W2 = np.random.randn(K,K)
b2 = np.random.randn(K)
V = np.random.randn(K,10)
c = np.random.randn(10)

error = []
classification = []
num = []
for i in range(10000):

    Z1 = feedForward(Xtrain.dot(W1) + b1)
    Z2 = feedForward(Z1.dot(W2) + b2)
    Y = softmax(Z2.dot(V) + c)

    dJ = Ytrain - Y
    dSZ1 = Z1 * (1 - Z1)
    dSZ2 = Z2 * (1 - Z2)

    V += learning_rate * Z2.T.dot(dJ)
    c += learning_rate * dJ.sum(axis=0)

    dJVt = dJ.dot(V.T)

    W2 += learning_rate * Z1.T.dot(dJVt * dSZ2)
    b2 += learning_rate * (dJVt * dSZ2).sum(axis=0)
    W1 += learning_rate * Xtrain.T.dot(dJVt * dSZ2 * dSZ1)
    b1 += learning_rate * (dJVt * dSZ2 * dSZ1).sum(axis=0)

    J = cross_entropy(Ytrain, Y)

    if i % 10 == 0:
        error.append(J)
        classification.append(np.round(100 * classification_rate(Ytrain, Y)))
        print(i , np.round(J,2),
            "| Target =", np.argmax(Ytrain[i]),
            "| Prediction =", np.argmax(Y[i]),
            "| Match =", np.argmax(Ytrain[i]) == np.argmax(Y[i]))

legend, = plt.plot(error, label='train cost')
plt.legend([legend])
plt.show()

legend, = plt.plot(classification, label='classification_rate')
plt.legend([legend])
plt.show()

print('Final classification_rate: ' + str(np.round(100 * classification_rate(Ytrain, Y),2)) + '%')
