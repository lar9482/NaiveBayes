from model.Model import Model
from joblib import Parallel, delayed

from multiprocessing import Process, Manager

import numpy as np
import math

class MultinomialNB(Model):

    def __init__(self, numVocab, numClasses, k = 0, fitParallel = True, fitJobs = 20):
        """
            @param numVocab: Integer
            The number of vocabulary words in the entire dataset

            @param numClasses: Integer
            The number of output classes possible

            @param k: Integer
            The prediction constant for Laplace smoothing

            @param fitParallel: Bool
            A flag to determine if fit(X, Y) should be ran in parallel

            @param fitJobs: Integer
            The number of subprocesses to help compute fitting each vocab word 
            with samples in the dataset

        """
        self.numVocab = numVocab
        self.numClasses = numClasses
        self.k = k
        self.fitJobs = fitJobs
        self.fitParallel = fitParallel

        # Sentienel character to indicate the end of a sample instance.
        self.terminateVocab = -1
        
        # Referenced by probClasses[i] where i is a class
        self.probClasses = {classNumber: 0 for classNumber in range(0, self.numClasses)}

        # Referenced by probVocabGivenClass[i][j] where i is a vocab word and j is a class
        self.probVocabGivenClass = {
            feature: {
                classNumber: 0 for classNumber in range(0, self.numClasses)
            } for feature in range(0, self.numVocab)
        }
    
    def fit(self, X, Y):
        lengthOfX = np.zeros((len(X), 1), dtype=int)
        for i in range(0, len(X)):
            lengthOfX[i] = self.findLengthOfDataInstance(X, i)

        for classOutput in range(0, self.numClasses):
            if (self.fitParallel):
                self.fitProbVocabGivenClass_Parallel(classOutput, lengthOfX, X, Y)
            else:
                self.fitProbVocabGivenClass_Sequential(classOutput, lengthOfX, X, Y)  
            
            self.fitClassOutput(classOutput, Y)
            print("Fitted class {0}".format(classOutput))
    
    def fitProbVocabGivenClass_Parallel(self, classOutput, lengthOfX, X, Y):
        """
            Instances of X,Y are processed in parallel.
            This is done be dividing the number of vocab words with the number of 
            samples in the dataset proportationally with self.fitJobs

            Thus, the number of vocabAndClassMatches and classMatches are computed in parallel.
        """
        intervals_DataAndVocab = self.__divideDatasetAndVocab(X)       
        sharedMatchesPerVocab = {}
        with Manager() as manager:
            allProcesses = []
            matchesPerVocabLock = manager.Lock()
            sharedMatchesPerVocabProxy = manager.dict()
            intervalSize = (len(intervals_DataAndVocab)) // self.fitJobs
            for i in range(0, self.fitJobs):
                unitsOfWork = intervals_DataAndVocab[i*intervalSize:(i+1)*intervalSize]
                allProcesses.append(Process(
                    target=self.__fitProbVocabGivenClass_Subprocess,
                    args=(
                        unitsOfWork, classOutput, lengthOfX, X, Y,
                        sharedMatchesPerVocabProxy, matchesPerVocabLock
                    )
                ))
            for process in allProcesses:
                process.start()
            for process in allProcesses:
                process.join()
            
            sharedMatchesPerVocab = dict(sharedMatchesPerVocabProxy)

        for vocab in list(sharedMatchesPerVocab.keys()):
            allVocabAndClassMatches = sharedMatchesPerVocab[vocab][0]
            allClassMatches = sharedMatchesPerVocab[vocab][1]
            PVocabGivenClass = (
                (allVocabAndClassMatches / allClassMatches) if self.k == 0
                else (
                    (allVocabAndClassMatches + self.k) /
                    (allClassMatches + self.k*(self.numVocab))
                )
            )
            self.probVocabGivenClass[vocab][classOutput] = PVocabGivenClass
            print("Parallel: Finishing vocab feature {0} with class {1}.".format(vocab, classOutput))

    def __divideDatasetAndVocab(self, X):
        intervalX = np.arange(0, len(X) + 1, (len(X)) // self.fitJobs)
        intervalVocab = np.arange(0, self.numVocab+1, (self.numVocab) // self.fitJobs)

        #Making sure the end of the intervals are the actual last elements of X or the dictionary.
        intervalX[len(intervalX)-1] = len(X)
        intervalVocab[len(intervalVocab)-1] = self.numVocab

        intervalXAndVocab = []
        for i in range(0, len(intervalX)-1):
            for j in range(0, len(intervalVocab)-1):
                startX = intervalX[i]
                endX = intervalX[i+1]
                startVocab = intervalVocab[j]
                endVocab = intervalVocab[j+1]
                intervalXAndVocab.append((startX, endX, startVocab, endVocab))

        return intervalXAndVocab
    
    def __fitProbVocabGivenClass_Subprocess(self, 
        unitsOfWork, classOutput, lengthOfX, X, Y,
        sharedMatchesPerVocabProxy, matchesPerVocabLock
    ):
        """
            Given every unit of (startX, endX, startVocab, endVocab), or the bounds of the dataset 
            and vocabulary dictionary,
            all jobs are computed and the numbers are aggregated.

            @returns: dict(vocab: (vocabAndClassMatch, classMatch))
        """
        subProcessMatchesPerVocab = {}
        for unit in unitsOfWork:
            startX = unit[0]
            endX = unit[1]
            startVocab = unit[2]
            endVocab = unit[3]
            localMatchesPerVocab = self.__fitProbVocabGivenClass_SubprocessJob(
                startX, endX, startVocab, endVocab,
                classOutput, lengthOfX, X, Y
            )

            for vocab in list(localMatchesPerVocab.keys()):
                localVocabAndClassMatch = localMatchesPerVocab[vocab][0]
                localClassMatches = localMatchesPerVocab[vocab][1]
                if (subProcessMatchesPerVocab.get(vocab) == None):
                    subProcessMatchesPerVocab[vocab] = [localVocabAndClassMatch, localClassMatches]
                else:
                    subProcessMatchesPerVocab[vocab][0] += localVocabAndClassMatch
                    subProcessMatchesPerVocab[vocab][1] += localClassMatches
        
        matchesPerVocabLock.acquire()
        for vocab in list(subProcessMatchesPerVocab.keys()):
            subProcessVocabAndClassMatches = subProcessMatchesPerVocab[vocab][0]
            subProcessClassMatches = subProcessMatchesPerVocab[vocab][1]
            if (sharedMatchesPerVocabProxy.get(vocab) == None):
                sharedMatchesPerVocabProxy[vocab] = [subProcessVocabAndClassMatches, subProcessClassMatches]
            else:
                sharedMatchesPerVocabProxy[vocab][0] += subProcessVocabAndClassMatches
                sharedMatchesPerVocabProxy[vocab][1] += subProcessClassMatches
        matchesPerVocabLock.release()

   
    def __fitProbVocabGivenClass_SubprocessJob(self, startX, endX, startVocab, endVocab,
            classOutput, lengthOfX, X, Y
        ):
        """
            Given the bounds of the dataset and the vocabulary dictionary,
            The job, orthe number of vocabAndClassMatches and classMatches, is computed.

            @returns: dict(vocab: (vocabAndClassMatch, classMatch))
        """
        localMatchesPerVocab = {}
        for vocab in range(startVocab, endVocab):
            localVocabAndClassMatch = 0
            localClassMatch = 0
            for i in range(startX, endX):
                lengthOfInstance = lengthOfX[i][0]
                for j in range(0, lengthOfInstance):
                    if (X[i][j] == vocab and Y[i] == classOutput):
                        localVocabAndClassMatch += 1
            
                if (Y[i] == classOutput):
                    localClassMatch += lengthOfInstance
                
            if (localMatchesPerVocab.get(vocab) == None):
                localMatchesPerVocab[vocab] = [localVocabAndClassMatch, localClassMatch]
            else:
                localMatchesPerVocab[vocab][0] += localVocabAndClassMatch
                localMatchesPerVocab[vocab][1] += localClassMatch

            print("Parallel: Fitted vocab feature {0} with class {1} from samples ({2},{3})".format(vocab, classOutput,startX,endX))
        return localMatchesPerVocab

    def fitProbVocabGivenClass_Sequential(self, classOutput, lengthOfX, X, Y):
        for vocab in range(0, self.numVocab):
            vocabAndClassMatch = 0
            classMatch = 0
            
            for i in range(0, len(X)):
                lengthOfInstance = lengthOfX[i][0]
                for j in range(0, lengthOfInstance):
                    if (X[i][j] == vocab and Y[i] == classOutput):
                        vocabAndClassMatch += 1
            
                if (Y[i] == classOutput):
                    classMatch += lengthOfInstance

            PVocabGivenClass = (
                (vocabAndClassMatch / classMatch) if self.k == 0
                else (
                    (vocabAndClassMatch + self.k) /
                    (classMatch + self.k*(self.numVocab))
                )
            )
        
            self.probVocabGivenClass[vocab][classOutput] = PVocabGivenClass
            print("Serial: Fitted vocab feature {0} with class {1}".format(vocab, classOutput))

    def findLengthOfDataInstance(self, X, i):
        d = 0
        for j in range(0, len(X[i])):
            if (X[i][j] == self.terminateVocab):
                break
            d += 1
        return d
    
    def fitClassOutput(self, classOutput, Y):
        classMatch = 0
        numSamples = len(Y)
        for i in range(0, numSamples):
            if (Y[i] == classOutput):
                classMatch += 1

        PClass = (
            classMatch / numSamples if self.k == 0 
            else (
                (classMatch + 1) / (numSamples + self.k)
            )
        )
        self.probClasses[classOutput] = PClass
    
    def classify(self, X):
        Y = np.zeros((len(X), 1), dtype=int)
        for i in range(0, len(X)):
            maxPClassGivenSample = -1
            maxClass = -1
            sample = X[i]

            #The normalizing constant
            PSampleGivenAllClasses = self.PSampleGivenAllClasses(sample)

            for classOutput in range(0, self.numClasses):
                PSampleGivenClass = math.log(
                    self.probClasses[classOutput]
                )
                for vocabInSample in sample:
                    if (vocabInSample == self.terminateVocab):
                        break
                    prob = self.probVocabGivenClass[vocabInSample][classOutput]
                    if (prob != 0):
                        PSampleGivenClass += math.log(prob)
                    else:
                        PSampleGivenClass = 0
                        break

                PClassGivenSample = (PSampleGivenClass / PSampleGivenAllClasses)

                if (PClassGivenSample > maxPClassGivenSample):
                    maxPClassGivenSample = PClassGivenSample
                    maxClass = classOutput
            
            Y[i] = maxClass
        
        return Y

    def PSampleGivenAllClasses(self, sample):
        PSample = 0
        for classOutput in range(0, self.numClasses):
            sumProb = math.log(
                self.probClasses[classOutput]
            )

            for vocabInSample in sample:
                if (vocabInSample == self.terminateVocab):
                    break
                prob = self.probVocabGivenClass[vocabInSample][classOutput]
                if (prob != 0):
                    sumProb += math.log(prob)
                else:
                    sumProb = 0
                    break
            
            PSample += sumProb
        
        return PSample