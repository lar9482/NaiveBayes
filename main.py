from preprocess.getMNIST import getMNIST
from preprocess.getPolitical import getPolSentences
def main():
    # (X, Y) = getMNIST()
    getPolSentences()
    print("Hello World")
    
if __name__ == '__main__':
    main()