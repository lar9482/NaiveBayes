from model.Model import Model

import numpy as np
import math

class MultinomialNB(Model):

    def __init__(self, numVocab, numClasses, k = 0):
        """
            @param numVocab: Integer
            The number of vocabulary words in the entire dataset

            @param numClasses: Integer
            The number of output classes possible

            @param k: Integer
            The prediction constant for Laplace smoothing
        """
        self.numVocab = numVocab
        self.numClasses = numClasses
        self.k = k

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
        for classOutput in range(0, self.numClasses):
            for vocab in range(0, self.numVocab):
                self.fitProbVocabGivenClass(vocab, classOutput, X, Y)
                print("Fitted vocab feature {0} with class {1}".format(vocab, classOutput))

            self.fitClassOutput(classOutput, Y)
            print("Fitted class {0}".format(classOutput))

    def fitProbVocabGivenClass(self, vocab, classOutput, X, Y):
        vocabAndClassMatch = 0
        classMatch = 0
        for i in range(0, len(X)):
            lengthOfInstance = self.findLengthOfDataInstance(X, i)

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
                PSampleGivenClass = abs(math.log(
                    self.probClasses[classOutput]
                ))
                for vocabInSample in sample:
                    if (vocabInSample == self.terminateVocab):
                        break
                    prob = self.probVocabGivenClass[vocabInSample][classOutput]
                    if (prob != 0):
                        PSampleGivenClass += abs(
                            math.log(prob)
                        )
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
            sumProb = abs(math.log(
                self.probClasses[classOutput]
            ))

            for vocabInSample in sample:
                if (vocabInSample == self.terminateVocab):
                    break
                prob = self.probVocabGivenClass[vocabInSample][classOutput]
                if (prob != 0):
                    sumProb += abs(
                        math.log(prob)
                    )
                else:
                    sumProb = 0
                    break
            
            PSample += sumProb
        
        return PSample