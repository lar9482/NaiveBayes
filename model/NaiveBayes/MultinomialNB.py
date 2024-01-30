from model.Model import Model

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

            self.fitClassOutput(classOutput, Y)

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
        for j in range(0, range(X[i])):
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
        pass