import numpy as np

def getPoliticialSentences(numInstances = 2500, booleanize = True):
    numInstancesLiberal = numInstances // 2
    numInstancesConservative = numInstances - numInstancesLiberal

    liberalSentences = getSentences(numInstancesLiberal, True)
    conservativeSentences = getSentences(numInstancesConservative, False)

    vocab = getVocabulary(liberalSentences, conservativeSentences)

    X = np.zeros((len(liberalSentences) + len(conservativeSentences), len(vocab)), dtype=float)
    Y = np.zeros((len(liberalSentences) + len(conservativeSentences), 1), dtype=float)

    
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