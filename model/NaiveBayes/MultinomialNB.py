from Model.Model import Model

class MultinomialNB(Model):

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

        # Sentienel character to indicate the end of a sample instance.
        self.terminateVocab = -1
        
        # Referenced by probClasses[i] where i is a class
        self.probClasses = {classNumber: 0 for classNumber in range(0, self.numClasses)}

        # Referenced by probVocabGivenClass[i][j] where i is a vocab word and j is a class
        self.probVocabGivenClass = {
            feature: {
                classNumber: 0 for classNumber in range(0, self.numClasses)
            } for feature in range(0, self.numFeatures)
        }
    
    def fit(self, X, Y):
        pass

    def classify(self, X):
        pass