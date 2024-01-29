from preprocess.getMNIST import getMNIST_Bernoulli, getMNIST_Multinomial
from preprocess.getPolitical import getPolSentences_Bernoulli, getPolSentences_Multinomial
from NFold import NFold

from model.NaiveBayes.BernoulliNB import BernoulliNB

def main():
    (X, Y) = getMNIST_Bernoulli(0, 1000)
    # (X, Y) = getPolSentences_Bernouli()
    # (X, Y) = getPolSentences_Multinomial()
    model = BernoulliNB(len(X[0]), 10, 2)
    model.fit(X, Y)
    newY = model.classify(X)
    for i in range(0, len(newY)):
        print(newY[i])
    # NFold(X, Y)
    
if __name__ == '__main__':
    main()