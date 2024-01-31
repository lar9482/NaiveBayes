from preprocess.getMNIST import getMNIST_Bernoulli, getMNIST_Multinomial
from preprocess.getPolitical import getPolSentences_Bernoulli, getPolSentences_Multinomial
from preprocess.getMovies import getMovieReviews_Bernoulli, getMovieReviews_Multinomial
from preprocess.shuffle import shuffleDataset
from NFold import NFold

from model.NaiveBayes.BernoulliNB import BernoulliNB
from model.NaiveBayes.MultinomialNB import MultinomialNB

import pandas as pd

from multiprocessing import Process, Manager

kOptions = [0, 1, 25, 5, 50, 100, 1000]
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
    DF = pd.DataFrame(columns=['k', 'trainAvg', 'testAvg'])

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
    DF = pd.DataFrame(columns=['k', 'trainAvg', 'testAvg'])

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
    
    index = 0
    for kTrainTestAcc in sharedList:
        k = kTrainTestAcc[0]
        trainAcc = kTrainTestAcc[1]
        testAcc = kTrainTestAcc[2]

        DF.loc[index] = [k, trainAcc, testAcc]
        index += 1

    DF.to_csv(fileName)

def main():
    (X, Y) = getMovieReviews_Bernoulli(2500)
    runBernoulliConfig('MoviesBernoulli.csv', 2, X, Y, 1)

    (X, Y, numVocab, numClasses) = getMovieReviews_Multinomial(2500)
    runMultinomialConfig('MoviesMultinomial.csv', numVocab, numClasses, X, Y, 1)
    # (X, Y) = getPolSentences_Bernoulli(1000)
    # (X, Y) = shuffleDataset(X, Y)
    # model = BernoulliNB(len(X[0]), 2, 50)
    # (trainAcc, testAcc) = NFold(X, Y, model)
    # print(trainAcc, testAcc)

    # (X, Y, numVocab, numClasses) = getMNIST_Multinomial(0, 250)
    # (X, Y) = shuffleDataset(X, Y)
    # multiNB_pol = MultinomialNB(numVocab, numClasses, 0)
    # (trainAcc, testAcc) = NFold(X, Y, multiNB_pol)
    # print(trainAcc, testAcc)
    # print()
    
if __name__ == '__main__':
    main()