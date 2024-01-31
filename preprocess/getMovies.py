import pandas as pd
import numpy as np
import csv
import re

def getMovieReviews_Bernoulli(numInstances = 100):
    (positiveSentences, negativeSentences) = getSentences(numInstances)
    vocab = getVocabulary(positiveSentences, negativeSentences)

    X = np.zeros((len(positiveSentences) + len(negativeSentences), len(vocab)), dtype=int)
    Y = np.zeros((len(positiveSentences) + len(negativeSentences), 1), dtype=float)
    
    for i in range(0, len(positiveSentences)):
        words = list(set(
            positiveSentences[i].split()
        ))
        
        for word in words:
            X[i][vocab.index(word)] = 1
            Y[i] = 0
        print(i)
    for i in range(0, len(negativeSentences)):
        words = list(set(
            negativeSentences[i].split()
        ))
        
        for word in words:
            X[i + len(positiveSentences)][vocab.index(word)] = 1
            Y[i + len(positiveSentences)] = 1
        print(i)
    return (X, Y)

def getMovieReviews_Multinomial(numInstances = 100):
    (positiveSentences, negativeSentences) = getSentences(numInstances)
    vocab = getVocabulary(positiveSentences, negativeSentences)

    X = np.zeros((len(positiveSentences) + len(negativeSentences), len(vocab)), dtype=float)
    Y = np.zeros((len(positiveSentences) + len(negativeSentences), 1), dtype=float)
    for i in range(0, len(positiveSentences)):
        words = list(positiveSentences[i].split())
        
        for j in range(0, len(vocab)):
            if (j < (len(words))):
                X[i][j] = vocab.index(words[j])
            else:
                X[i][j] = -1

            Y[i] = 0

    for i in range(0, len(negativeSentences)):
        words = list(negativeSentences[i].split())
        
        for j in range(0, len(vocab)):
            if (j < (len(words))):
                X[i + len(positiveSentences)][j] = vocab.index(words[j])
            else:
                X[i + len(positiveSentences)][j] = -1
                
            Y[i + len(positiveSentences)] = 1
    
    return (X, Y, len(vocab), 2)


def getSentences(numInstances):
    df = pd.read_csv('./dataset/movies/IMDB_Dataset.csv')
    reviews = df['review']
    ratings = df['sentiment']

    positiveSentences = []
    negativeSentences = []

    for i in range(0, numInstances):
        if (numInstances > len(reviews)):
            break
            
        preprocessedSentence = preprocessSentence(reviews[i])
        if (ratings[i] == 'positive'):
            positiveSentences.append(preprocessedSentence)
        else:
            negativeSentences.append(preprocessedSentence)
    
    return (positiveSentences, negativeSentences)

def preprocessSentence(sentence):
    sentence = re.sub('[,\.!?:()"]', '', sentence)
    sentence = re.sub('<.*?>', ' ', sentence)
    sentence = re.sub('http\S+', ' ', sentence)
    sentence = re.sub('[^a-zA-Z0-9]', ' ', sentence)
    sentence = re.sub('\s+', ' ', sentence)
    return sentence.lower().strip()

def getVocabulary(positiveSentences, negativeSentences):
    positiveWords = ' '.join(positiveSentences).split()
    negativeWords = ' '.join(negativeSentences).split()

    allWords = positiveWords + negativeWords
    uniqueWords = list(set(allWords))
    uniqueWords = sorted(uniqueWords)
    
    return uniqueWords