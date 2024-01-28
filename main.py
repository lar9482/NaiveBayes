from preprocess.getMNIST import getMNIST
from preprocess.getPolitical import getPolSentences_Bernouli
def main():
    # (X, Y) = getMNIST()
    (X, Y) = getPolSentences_Bernouli()
    print("Hello World")
    
if __name__ == '__main__':
    main()