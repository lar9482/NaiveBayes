class Model:

    def fit(self, X, Y):
        """
            Implementation of the training model.
            @param X: np.array of (n_samples, n_features)
            @param Y: np.array of (n_samples, 1) 
        """
        pass

    def classify(self, X):
        """
            Implementation of the classify model
            @param X: np.array of (n_samples, n_features)
        """
        pass

    def evaluate(self, X, Y):
        """
            Implementation of evaluation for the number of correct guesses.
            @param X: np.array of (n_samples, n_features)
            @param Y: np.array of (n_samples, 1) 
        """
        correctGuesses = 0
        predictedY = self.classify(X)
        for i in range(0, len(Y)):
            if (predictedY[i] == Y[i]):
                correctGuesses += 1

        return correctGuesses / len(Y)