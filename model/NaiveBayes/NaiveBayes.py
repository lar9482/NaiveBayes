from model.Model import Model

class NaiveBayes(Model):
    def __init__(self, numFeatures, numClasses):
        self.numFeatures = numFeatures
        self.numClasses = numClasses

        # Referenced by probClasses[i] where i is a class
        self.probClasses = {classNumber: 0 for classNumber in range(0, self.numClasses)}

        # Referenced by probFeatureGivenClass[i][j] where i is a feature and j is a class
        self.probFeatureGivenClass = {
            feature: {
                classNumber: 0 for classNumber in range(0, self.numClasses)
            } for feature in range(0, self.numFeatures)
        }