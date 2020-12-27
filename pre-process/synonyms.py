import nltk
from nltk.corpus import wordnet   #Import wordnet from the NLTK
from nltk.stem import WordNetLemmatizer
import string

anger_words = []
disgust_words = []
fear_words = []
joy_words = []
sadness_words = []
surprise_words = []

negative_words = []
positive_words = []
stop_words = []

def load_datasets():
    global anger_words
    global disgust_words
    global fear_words
    global joy_words
    global sadness_words
    global surprise_words
    global negative_words
    global positive_words
    global stop_words
    
    #read stop words
    f = open('datasets/sentiments/stopwords.txt', 'r') 
    stop_words = [word for line in f for word in line.split()]
    f.close()
    

    #read negative words
    f = open('datasets/sentiments/negative-words.txt', 'r') 
    negative_words = [word for line in f for word in line.split()]
    f.close()

    #read positive words
    f = open('datasets/sentiments/positive-words.txt', 'r') 
    positive_words = [word for line in f for word in line.split()]
    f.close()

    #read anger words
    f = open('datasets/WordNetAffectEmotionLists/anger.txt', 'r') 
    anger_words = [word for line in f for word in line.split()]
    f.close()

    #read disgust words
    f = open('datasets/WordNetAffectEmotionLists/disgust.txt', 'r') 
    disgust_words = [word for line in f for word in line.split()]
    f.close()

    #read fear words
    f = open('datasets/WordNetAffectEmotionLists/fear.txt', 'r') 
    fear_words = [word for line in f for word in line.split()]
    f.close()

    #read joy words
    f = open('datasets/WordNetAffectEmotionLists/joy.txt', 'r') 
    joy_words = [word for line in f for word in line.split()]
    f.close()

    #read sadness words
    f = open('datasets/WordNetAffectEmotionLists/sadness.txt', 'r') 
    sadness_words = [word for line in f for word in line.split()]
    f.close()

    #read surprise words
    f = open('datasets/WordNetAffectEmotionLists/surprise.txt', 'r') 
    surprise_words = [word for line in f for word in line.split()]
    f.close()
    
load_datasets()



def writeInFile(file, l):
    f = open(file, "a")
    for word in l:
        f.write(word)
        f.write("\n")
    f.close()

def synonym(file,words):
    synonyms_words = []
    for word in words:
        synonyms_words.append(word)
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                synonyms_words.append(lemma.name())
    synonyms_words = set(synonyms_words)
    writeInFile(file,synonyms_words)

synonym("datasets/WordNetAffectEmotionLists/supriseSynonyms.txt",surprise_words)
synonym("datasets/WordNetAffectEmotionLists/angerSynonyms.txt",anger_words)
synonym("datasets/WordNetAffectEmotionLists/joySynonyms.txt",joy_words)
synonym("datasets/WordNetAffectEmotionLists/sadnessSynonyms.txt",sadness_words)
synonym("datasets/WordNetAffectEmotionLists/fearSynonyms.txt",fear_words)
synonym("datasets/WordNetAffectEmotionLists/disgustSynonyms.txt",disgust_words)

synonym("datasets/sentiments/stopwordsSynonyms.txt",stop_words)
synonym("datasets/sentiments/positivewordsSynonyms.txt",positive_words)
synonym("datasets/sentiments/negativewordsSynonyms.txt",negative_words)







