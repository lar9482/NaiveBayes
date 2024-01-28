from preprocess.getMNIST import getMNIST_Bernoulli, getMNIST_Multinomial
from preprocess.getPolitical import getPolSentences_Bernoulli, getPolSentences_Multinomial
from NFold import NFold

def main():
    (X, Y) = getMNIST_Multinomial(0, 10)
    # (X, Y) = getPolSentences_Bernouli()
    # (X, Y) = getPolSentences_Multinomial()
    NFold(X, Y)
    
if __name__ == '__main__':
    main()