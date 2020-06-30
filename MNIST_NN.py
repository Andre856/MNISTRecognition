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

def derivative_v(Z, T, Y):
    return Z.T.dot(T - Y)

def derivative_c(T, Y):
    return (T - Y).sum(axis=0)

def derivative_w2(Z1, Z2, T, Y, V):
    dZ2 = (T - Y).dot(V.T) * Z2 * (1 - Z2)
    return Z1.T.dot(dZ2)

def derivative_b2(T, Y, V, Z2):
    return ((T - Y).dot(V.T) * Z2 * (1 - Z2)).sum(axis=0)

def derivative_w1(X, Z1, Z2, T, Y, V):
    dZ1 = (T - Y).dot(V.T) * Z2 * (1 - Z2) * Z1 * (1 - Z1)
    return X.T.dot(dZ1)

def derivative_b1(T, Y, V, Z2, Z1):
    return ((T - Y).dot(V.T) * Z2 * (1 - Z2) * Z1 * (1 - Z1)).sum(axis=0)

def classification_rate(T, Y):
    return np.mean(np.argmax(T, axis=1) == np.argmax(Y, axis=1))


learning_rate = 0.000025

K = 25 # K = Number of neurons in each layer
W1 = np.random.randn(Xtrain.shape[1], K)
b1 = np.random.randn(K)
W2 = np.random.randn(K,K)
b2 = np.random.randn(K)
V = np.random.randn(K,10)
c = np.random.randn(10)

error = []
classification = []
num = []
for i in range(1000):

    Z1 = feedForward(Xtrain.dot(W1) + b1)
    Z2 = feedForward(Z1.dot(W2) + b2)
    Y = softmax(Z2.dot(V) + c)

    V += learning_rate * derivative_v(Z2, Ytrain, Y)
    c += learning_rate * derivative_c(Ytrain, Y)
    W2 += learning_rate * derivative_w2(Z1, Z2, Ytrain, Y, V)
    b2 += learning_rate * derivative_b2(Ytrain, Y, V, Z2)
    W1 += learning_rate * derivative_w1(Xtrain, Z1, Z2, Ytrain, Y, V)
    b1 += learning_rate * derivative_b1(Ytrain, Y, V, Z2, Z1)

    J = cross_entropy(Ytrain, Y)

    if i % 10 == 0:
        error.append(J)
        classification.append(np.round(100 * classification_rate(Ytrain, Y)))
        print(i , np.round(J,2), "| Target =", np.argmax(Ytrain[i]), "| Prediction =", np.argmax(Y[i]), "| Match =", np.argmax(Ytrain[i]) == np.argmax(Y[i]))
        if i > 9000 and np.argmax(Ytrain[i]) != np.argmax(Y[i]):
            num.append(i)


legend1, = plt.plot(error, label='train cost')
plt.legend([legend1])
plt.show()

legend1, = plt.plot(classification, label='classification_rate')
plt.legend([legend1])
plt.show()

print('Final classification_rate: ' + str(np.round(100 * classification_rate(Ytrain, Y),2)) + '%')

for i in num:
    digit = image[i]
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()

