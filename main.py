from preprocess.getMNIST import getMNIST_Bernoulli, getMNIST_Multinomial
from preprocess.getPolitical import getPolSentences_Bernoulli, getPolSentences_Multinomial
from NFold import NFold

from model.NaiveBayes.BernoulliNB import BernoulliNB

def main():
    (X, Y) = getMNIST_Bernoulli(0, 100)
    # (X, Y) = getPolSentences_Bernouli()
    # (X, Y) = getPolSentences_Multinomial()
    model = BernoulliNB(len(X[0]), 10, 1)
    model.fit(X, Y)
    NFold(X, Y)
    
if __name__ == '__main__':
    main()