import numpy as np
import csv

def getMNIST_Bernoulli(threshold = 0, numInstances = 1000):
    """
        X:
        If the 1st and 3rd pixels are on, then the image is encoded as 
        [1, 0, 1, 0,,,0]

        Y:
        The label of 0 through 9

        @param threshold: Integer
        The threshold of intensity for when a pixel should considered on or off.

        @param numInstances: Integer
        The number of samples to get from the dataset

        return (X, Y):
        The encoded data
    """
    trainFile = open('./dataset/digits/train.csv', 'r')
    rawTrainData = list(csv.reader(trainFile))

    #Removing the first row. It just describes the labelling.
    rawTrainData = rawTrainData[1:len(rawTrainData)]
    matrix = np.array(rawTrainData, dtype=float)
    
    X = np.zeros((numInstances, 784), dtype=float)
    Y = np.zeros((numInstances, 1), dtype=float)
    
    if (numInstances < 0 or numInstances > len(matrix)):
        raise Exception('getMNIST: numInstances must be less than {0} and greater than 0'.format(len(matrix)))

    for i in range(0, numInstances):
        Y[i] = matrix[i][0]
        for j in range(1, len(matrix[i])):
            if (matrix[i][j] > threshold):
                X[i][j-1] = 1
            else:
                X[i][j-1] = 0

    return (X, Y)

def getMNIST_Multinomial(threshold = 0, numInstances = 100):
    """
        X:
        If the 1st and 3rd pixels are on, then the image is encoded as 
        [1, 3, -1,,,-1], with -1 placeholders to indicate the end of the number of pixels.
        Why? Remember that the number of pixels per image is different.

        Y:
        The label of 0 through 9

        @param threshold: Integer
        The threshold of intensity for when a pixel should considered on or off.

        @param numInstances: Integer
        The number of samples to get from the dataset

        return (X, Y):
        The encoded data
    """
    trainFile = open('./dataset/digits/train.csv', 'r')
    rawTrainData = list(csv.reader(trainFile))

    #Removing the first row. It just describes the labelling.
    rawTrainData = rawTrainData[1:len(rawTrainData)]
    matrix = np.array(rawTrainData, dtype=float)
    
    X = np.zeros((numInstances, 784), dtype=float)
    Y = np.zeros((numInstances, 1), dtype=float)
    
    if (numInstances < 0 or numInstances > len(matrix)):
        raise Exception('getMNIST: numInstances must be less than {0} and greater than 0'.format(len(matrix)))

    for i in range(0, numInstances):
        Y[i] = matrix[i][0]
        pixelsOn = []
        for j in range(1, len(matrix[i])):
            if (matrix[i][j] > threshold):
                pixelsOn.append(j-1)
        
        index = 0
        for j in range(0, len(X[i])):
            if (index < len(pixelsOn)):
                X[i][j] = pixelsOn[index]
                index += 1
            else:
                X[i][j] = -1
    return (X, Y, 784, 10)