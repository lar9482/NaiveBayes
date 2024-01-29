from preprocess.getMNIST import getMNIST_Bernoulli, getMNIST_Multinomial
from preprocess.getPolitical import getPolSentences_Bernoulli, getPolSentences_Multinomial
from preprocess.shuffle import shuffleDataset
from NFold import NFold

from model.NaiveBayes.BernoulliNB import BernoulliNB

def main():
    (X, Y) = getPolSentences_Bernoulli(1000)
    (X, Y) = shuffleDataset(X, Y)
    model = BernoulliNB(len(X[0]), 2, 0)
    (trainAcc, testAcc) = NFold(X, Y, model)
    print(trainAcc, testAcc)
    
if __name__ == '__main__':
    main()