from model.Model import Model

import numpy as np
import math

class BernoulliNB(Model):

    def __init__(self, numFeatures, numClasses, k = 0):
        """
            @param numFeatures: Integer
            The number of features per sample

            @param numClasses: Integer
            The number of output classes possible

            @param k: Integer
            The prediction constant for Laplace smoothing
        """
        self.numFeatures = numFeatures
        self.numClasses = numClasses
        self.k = k

        # Referenced by probClasses[i] where i is a class
        self.probClasses = {classNumber: 0 for classNumber in range(0, self.numClasses)}

        # Referenced by probFeatureGivenClass[i][j] where i is a feature and j is a class
        self.probFeatureGivenClass = {
            feature: {
                classNumber: 0 for classNumber in range(0, self.numClasses)
            } for feature in range(0, self.numFeatures)
        }
    
    def fit(self, X, Y):
        for feature in range(0, self.numFeatures):
            for classOutput in range(0, self.numClasses):
                self.fitFeaturePerClassOutput(feature, classOutput, X, Y)

                print("Fitted feature {0} on class {1}".format(feature, classOutput))

        for classOutput in range(0, self.numClasses):
            self.fitClassOutput(classOutput, Y)

    def fitFeaturePerClassOutput(self, feature, classOutput, X, Y):
        featureAndClassMatch = 0
        classMatch = 0
        numSamples = len(X)

        for sample in range(0, numSamples):
            if (X[sample][feature] == 1 and Y[sample] == classOutput):
                featureAndClassMatch += 1
            
            if (Y[sample] == classOutput):
                classMatch += 1    

        self.probFeatureGivenClass[feature][classOutput] = (
            (featureAndClassMatch + self.k) / 
            (classMatch + self.k*(2))
        )

    def fitClassOutput(self, classOutput, Y):
        classMatch = 0
        numSamples = len(Y)
        for i in range(0, numSamples):
            if (Y[i] == classOutput):
                classMatch += 1

        self.probClasses[classOutput] = (
            classMatch / numSamples
        )

    def classify(self, X):
        Y = np.zeros((len(X), 1), dtype=float)

        for i in range(0, len(X)):
            maxPClassGivenSample = -1
            maxClass = -1
            sample = X[i]

            PSampleGivenAllClasses = self.PSampleGivenAllClasses(sample)
            for classOutput in range(0, self.numClasses):
                PSampleGivenClass = math.log(
                    self.probClasses[classOutput]
                )
                #IS THIS WRONG?????
                for feature in range(0, self.numFeatures):
                    prob = self.probFeatureGivenClass[feature][classOutput]
                    if (prob != 0):
                        PSampleGivenClass += (
                            math.log(prob) 
                            if sample[feature] == 1
                            else math.log(1 - prob)
                        )
                PClassGivenSample = (PSampleGivenClass / PSampleGivenAllClasses)

                if (maxPClassGivenSample < PClassGivenSample):
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

            #IS THIS WRONG????
            for feature in range(0, self.numFeatures):
                prob = self.probFeatureGivenClass[feature][classOutput]
                if (prob != 0):
                    sumProb += (
                        math.log(prob) 
                        if sample[feature] == 1
                        else math.log(1 - prob)
                    )
            
            PSample += sumProb
        
        return PSample