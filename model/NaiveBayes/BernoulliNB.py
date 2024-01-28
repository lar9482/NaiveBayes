from model.Model import Model

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

        for classOutput in range(0, self.numClasses):
            self.fitClassOutput(classOutput, Y)

    def fitFeaturePerClassOutput(self, feature, classOutput, X, Y):
        featureAndClassMatch = 0
        classMatch = 0
        numSamples = len(X)

        for i in range(0, numSamples):
            if (X[i][feature] == 1 and Y[i] == classOutput):
                featureAndClassMatch += 1
            
            if (Y[i] == classOutput):
                classMatch += 1

        self.probFeatureGivenClass[feature][classOutput] = (
            (featureAndClassMatch + self.k) / 
            (classMatch + self.k*(numSamples))
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
        pass