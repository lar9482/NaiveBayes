from preprocess.getMNIST import getMNIST_Bernouli, getMNIST_Multinomial
from preprocess.getPolitical import getPolSentences_Bernouli, getPolSentences_Multinomial
def main():
    (X, Y) = getMNIST_Multinomial()
    # (X, Y) = getPolSentences_Bernouli()
    # (X, Y) = getPolSentences_Multinomial()
    print("Hello World")
    
if __name__ == '__main__':
    main()