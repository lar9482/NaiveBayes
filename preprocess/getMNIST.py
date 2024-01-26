import numpy as np
import csv

def getMNIST(threshold = 0, numInstances = 1000):
    """
        Extracting booleanized data from MNIST

        @param threshold: Integer
        The threshold of intensity for when a pixel should considered on or off.

        @param numInstances: Integer
        The number of samples to get from the dataset
    """
    trainFile = open('./dataset/digits/train.csv', 'r')
    rawTrainData = list(csv.reader(trainFile))

    #Removing the first row. It just describes the labelling.
    rawTrainData = rawTrainData[1:len(rawTrainData)]
    trainMatrixData = np.array(rawTrainData, dtype=float)

    matrix = trainMatrixData
    X = np.zeros((numInstances, 784), dtype=float)
    Y = np.zeros((numInstances, 1), dtype=float)
    
    if (numInstances > len(matrix)):
        raise Exception('getMNIST: numInstances must be less than {0}'.format(len(matrix)))

    for i in range(0, numInstances):
        Y[i] = matrix[i][0]
        for j in range(1, len(matrix[i])):
            if (matrix[i][j] > threshold):
                X[i][j-1] = 1
            else:
                X[i][j-1] = 0

    return (X, Y)

    
        
