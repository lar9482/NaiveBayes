import numpy as np

def getPolSentences_Bernoulli(numInstances = 2):
    """
        Encoding Scheme:
        X:
        Given the sentence, "Tulsa University", with the words positioned as the 10th and 11th words in the dictionary(sorted),

        then the sentence encoded as [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,,,0] 
        with placeholder zeros for the rest of the dictionary.

        Y:
        Liberal is assigned 0
        Conservative is assigned 1.
    """
    numInstancesLiberal = (numInstances // 2) - 1
    numInstancesConservative = numInstances - numInstancesLiberal

    liberalSentences = getSentences(numInstancesLiberal, True)
    conservativeSentences = getSentences(numInstancesConservative, False)
    vocab = getVocabulary(liberalSentences, conservativeSentences)

    X = np.zeros((len(liberalSentences) + len(conservativeSentences), len(vocab)), dtype=float)
    Y = np.zeros((len(liberalSentences) + len(conservativeSentences), 1), dtype=float)

    for i in range(0, len(liberalSentences)):
        words = list(set(
            liberalSentences[i].split()
        ))
        
        for word in words:
            X[i][vocab.index(word)] = 1
            Y[i] = 0
    
    for i in range(0, len(conservativeSentences)):
        words = list(set(
            conservativeSentences[i].split()
        ))
        
        for word in words:
            X[i + len(liberalSentences)][vocab.index(word)] = 1
            Y[i + len(liberalSentences)] = 1
    
    return (X, Y)

def getPolSentences_Multinomial(numInstances = 2):
    """
        Encoding Scheme:
        X:
        Given the sentence, "Tulsa University", with the words positioned as the 10th and 11th words in the dictionary(sorted),
        then the sentence encoded as [10, 11, -1,,,,-1], with placeholder -1 to indicate end of the sentence. 
        Why? Remember that the length of each email is different.

        Y:
        Liberal is assigned 0
        Conservative is assigned 1.

        returns (X, Y, numVocab, numClasses)
    """
    numInstancesLiberal = (numInstances // 2) - 1
    numInstancesConservative = numInstances - numInstancesLiberal

    liberalSentences = getSentences(numInstancesLiberal, True)
    conservativeSentences = getSentences(numInstancesConservative, False)
    vocab = getVocabulary(liberalSentences, conservativeSentences)

    X = np.zeros((len(liberalSentences) + len(conservativeSentences), len(vocab)), dtype=int)
    Y = np.zeros((len(liberalSentences) + len(conservativeSentences), 1), dtype=int)

    for i in range(0, len(liberalSentences)):
        words = list(liberalSentences[i].split())
        
        for j in range(0, len(vocab)):
            if (j < (len(words))):
                X[i][j] = vocab.index(words[j])
            else:
                X[i][j] = -1

            Y[i] = 0

    for i in range(0, len(conservativeSentences)):
        words = list(conservativeSentences[i].split())
        
        for j in range(0, len(vocab)):
            if (j < (len(words))):
                X[i + len(liberalSentences)][j] = vocab.index(words[j])
            else:
                X[i + len(liberalSentences)][j] = -1
                
            Y[i + len(liberalSentences)] = 0
    
    return (X, Y, len(vocab), 2)

def getSentences(numInstances, isLiberal):
    filePath = (
        "./dataset/politicial/liberal.txt" if isLiberal 
        else "./dataset/politicial/conservative.txt"
    )

    with open(filePath) as f:
        sentences = []
        for line in f:
            if (len(sentences) > numInstances):
                break

            strippedSentence = line.replace('0\t', '').replace('\n', '')
            sentences.append(strippedSentence)
        
        return sentences
    
def getVocabulary(liberalSentences, conservativeSentences):
    liberalWords = ' '.join(liberalSentences).split()
    conservativeWords = ' '.join(conservativeSentences).split()

    allWords = liberalWords + conservativeWords
    uniqueWords = list(set(allWords))
    uniqueWords = sorted(uniqueWords)
    
    return uniqueWords