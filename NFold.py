import numpy as np

def NFold(X, Y, model, N = 5):
    if (len(X) != len(Y)):
        raise Exception('The number of samples in X and Y must be the same')
    
    trainAcc = 0
    testAcc = 0

    difference = int((len(X) - (len(X) % N)) / N)
    startIndex = 0
    endIndex = difference
    numIterations = 0

    while (endIndex < len(X)):
        trainX = np.delete(X, range(startIndex, endIndex), axis = 0)
        testX = X[startIndex:endIndex, :]
        trainY = np.delete(Y, range(startIndex, endIndex), axis = 0)
        testY = Y[startIndex:endIndex, :]
        
        #Do training and testing here
        model.fit(trainX, trainY)

        trainAcc += model.evaluate(trainX, trainY)
        testAcc += model.evaluate(testX, testY)

        startIndex = endIndex
        endIndex += difference
        numIterations += 1

    return (
        (trainAcc / numIterations), 
        (testAcc / numIterations)
    )