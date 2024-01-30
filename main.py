from preprocess.getMNIST import getMNIST, getMNIST_Multinomial
from preprocess.getPolitical import getPolSentences_Bernoulli, getPolSentences_Multinomial
from preprocess.shuffle import shuffleDataset
from NFold import NFold

from model.NaiveBayes.BernoulliNB import BernoulliNB
from model.NaiveBayes.MultinomialNB import MultinomialNB

def main():
    # (X, Y) = getPolSentences_Bernoulli(1000)
    # (X, Y) = shuffleDataset(X, Y)
    # model = BernoulliNB(len(X[0]), 2, 50)
    # (trainAcc, testAcc) = NFold(X, Y, model)
    # print(trainAcc, testAcc)

    (X, Y, numVocab, numClasses) = getMNIST_Multinomial(0, 250)
    (X, Y)
    (X, Y) = shuffleDataset(X, Y)
    multiNB_pol = MultinomialNB(numVocab, numClasses, 0)
    (trainAcc, testAcc) = NFold(X, Y, multiNB_pol)
    print(trainAcc, testAcc)
    print()
    
if __name__ == '__main__':
    main()