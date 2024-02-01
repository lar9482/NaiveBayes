from preprocess.getMNIST import getMNIST_Bernoulli, getMNIST_Multinomial
from preprocess.getPolitical import getPolSentences_Bernoulli, getPolSentences_Multinomial
from preprocess.getMovies import getMovieReviews_Bernoulli, getMovieReviews_Multinomial
from preprocess.shuffle import shuffleDataset
from NFold import NFold

from model.NaiveBayes.BernoulliNB import BernoulliNB
from model.NaiveBayes.MultinomialNB import MultinomialNB

import pandas as pd

from multiprocessing import Process, Manager

kOptions = [0, 1, 50, 100, 1000]
def runNFold_Bernoulli(X, Y, numClasses, k, configLock, sharedList):
    model = BernoulliNB(len(X[0]), numClasses, k)
    (trainAcc, testAcc) = NFold(X, Y, model)

    configLock.acquire()
    sharedList.append((k, trainAcc, testAcc))
    configLock.release()

def runNFold_Multinomial(X, Y, numVocab, numClasses, k, configLock, sharedList):
    multiNB = MultinomialNB(numVocab, numClasses, k)
    (trainAcc, testAcc) = NFold(X, Y, multiNB)

    configLock.acquire()
    sharedList.append((k, trainAcc, testAcc))
    configLock.release()

def runBernoulliConfig(fileName, numClasses, X, Y, numTimes):
    (X, Y) = shuffleDataset(X, Y)

    sharedList = []
    with Manager() as manager:
        allProcesses = []
        configLock = manager.Lock()
        sharedListProxy = manager.list()
        for k in kOptions:
            for _ in range(0, numTimes):
                allProcesses.append(Process(
                    target=runNFold_Bernoulli, 
                    args=(
                        X, Y, numClasses, k, configLock, sharedListProxy
                    )
                ))
        
        for process in allProcesses:
            process.start()

        for process in allProcesses:
            process.join()

        sharedList = list(sharedListProxy)
    
    DF = pd.DataFrame(columns=['k', 'trainAvg', 'testAvg'])
    index = 0
    for kTrainTestAcc in sharedList:
        k = kTrainTestAcc[0]
        trainAcc = kTrainTestAcc[1]
        testAcc = kTrainTestAcc[2]

        DF.loc[index] = [k, trainAcc, testAcc]
        index += 1

    DF.to_csv(fileName)

def runMultinomialConfig(fileName, numVocab, numClasses, X, Y, numTimes):
    (X, Y) = shuffleDataset(X, Y)

    sharedList = []
    with Manager() as manager:
        allProcesses = []
        configLock = manager.Lock()
        sharedListProxy = manager.list()
        for k in kOptions:
            for _ in range(0, numTimes):
                allProcesses.append(Process(
                    target=runNFold_Multinomial, 
                    args=(
                        X, Y, numVocab, numClasses, k, configLock, sharedListProxy
                    )
                ))
        
        for process in allProcesses:
            process.start()

        for process in allProcesses:
            process.join()

        sharedList = list(sharedListProxy)
    
    DF = pd.DataFrame(columns=['k', 'trainAvg', 'testAvg'])
    index = 0
    for kTrainTestAcc in sharedList:
        k = kTrainTestAcc[0]
        trainAcc = kTrainTestAcc[1]
        testAcc = kTrainTestAcc[2]

        DF.loc[index] = [k, trainAcc, testAcc]
        index += 1

    DF.to_csv(fileName)

def runMoviesBernoulli(numInstances, numTimes = 1):
    (X, Y) = getMovieReviews_Bernoulli(numInstances)
    runBernoulliConfig('MoviesBernoulli.csv', 2, X, Y, numTimes)

def runMoviesMultinomial(numInstances, numTimes = 1):
    (X, Y, numVocab, numClasses) = getMovieReviews_Multinomial(numInstances)
    runMultinomialConfig('MoviesMultinomial.csv', numVocab, numClasses, X, Y, numTimes)

def runMNISTV1(numInstances, numTimes = 1):
    (X, Y) = getMNIST_Bernoulli(0, numInstances)
    numVocabV1 = 2
    numClassesV1 = 10
    runMultinomialConfig('MNISTMultinomialV1.csv', numVocabV1, numClassesV1, X, Y, numTimes)

def runMNISTV2(numInstances, numTimes = 1):
    (X, Y, numVocabV2, numClassesV2) = getMNIST_Multinomial(0, numInstances)
    runMultinomialConfig('MNISTMultinomialV2.csv', numVocabV2, numClassesV2, X, Y, numTimes)

def main():
    numInstances = 5000
    numTimes = 1
    with Manager() as manager:
        allProcesses = []
        allProcesses.append(Process(
            target=runMoviesBernoulli, 
            args = (50000, numTimes)
        ))
        allProcesses.append(Process(
            target=runMoviesMultinomial, 
            args = (50000, numTimes)
        ))
        # allProcesses.append(Process(
        #     target=runMNISTV1, 
        #     args = (500, numTimes)
        # ))
        # allProcesses.append(Process(
        #     target=runMNISTV2, 
        #     args = (500, numTimes)
        # ))
        
        for process in allProcesses:
            process.start()

        for process in allProcesses:
            process.join()
    
if __name__ == '__main__':
    main()