import nltk
from nltk.stem import WordNetLemmatizer
import xml.dom.minidom
from xml.dom import minidom
import string

#list of phrases with ids
training_data = {}
test_data = {}

#emotions words
anger_words = []
disgust_words = []
fear_words = []
joy_words = []
sadness_words = []
surprise_words = []

#valence words
negative_words = []
positive_words = []

lemmatizer = WordNetLemmatizer()

#============================Load datasets============================

def load_data(file):
    f = open(file, 'r') 
    words = [word.lower() for line in f for word in line.split()]
    f.close()
    return words

def load_datasets_emotions():
    global anger_words
    global disgust_words
    global fear_words
    global joy_words
    global sadness_words
    global surprise_words
    
    anger_words = load_data('datasets/WordNetAffectEmotionLists/angerSynonyms.txt')
    disgust_words = load_data('datasets/WordNetAffectEmotionLists/disgustSynonyms.txt') 
    
    #read fear words
    fear_words= load_data('datasets/WordNetAffectEmotionLists/fearSynonyms.txt') 
    
    #read joy words
    joy_words = load_data('datasets/WordNetAffectEmotionLists/joySynonyms.txt') 
    
    #read sadness words
    sadness_words = load_data('datasets/WordNetAffectEmotionLists/sadnessSynonyms.txt') 
    
    #read surprise words
    surprise_words = load_data('datasets/WordNetAffectEmotionLists/surpriseSynonyms.txt')

def load_datasets_sentiments():
    global negative_words
    global positive_words
    #read negative words
    negative_words= load_data('datasets/SentimentWords/negative.txt') 

    #read positive words
    positive_words= load_data('datasets/SentimentWords/positive.txt')

def load_all_data():
    load_datasets_emotions()
    load_datasets_sentiments()

#============================Process phrases============================
def binaryResult(value, limit):
    if value < limit:
        return "0"
    else:
        return "1"

def preprocessPhrase(phrase):
    phrase = phrase.replace('\n', '')
    phrase = phrase.translate(str.maketrans('', '', string.punctuation)) #remove punctuation signs
    return phrase

#return -1 if negative 0 if neutral and 1 if positive
def processPhraseValence(phrase):
    negative_count = 0
    positive_count = 0
    words = nltk.word_tokenize(phrase)
    for word in words:
        word_lemmatized = lemmatizer.lemmatize(word)
        if word_lemmatized in positive_words or word in positive_words:
            positive_count += 1
        elif word_lemmatized in negative_words or word in negative_words:
            negative_count += 1

    pourcentageNeg = negative_count*100/len(words)
    pourcentagePos = positive_count*100/len(words)
    pourcentageNeu = 100 - pourcentageNeg - pourcentagePos

    if pourcentagePos > pourcentageNeg:
        return "1"
    return "-1"

#return a string with 0 or 1 in format: anger disgust fear joy sadness surprise
def processPhraseEmotions(phrase):
    anger_count = 0
    disgust_count = 0
    fear_count = 0
    joy_count = 0
    sadness_count = 0
    surprise_count = 0
    total_count = 0
    words = nltk.word_tokenize(phrase)
    for word in words:
        word_lemmatized = lemmatizer.lemmatize(word)
        if word_lemmatized in anger_words or word in anger_words :
            anger_count+=1
        elif word_lemmatized in disgust_words:
            disgust_count+=1
        elif word_lemmatized in fear_words:
            fear_count+=1
        elif word_lemmatized in joy_words:
            joy_count+=1
        elif word_lemmatized in sadness_words:
            sadness_count+=1
        elif word_lemmatized in surprise_words:
            surprise_count+=1  
        total_count+=1

    limit = 0.1 #if emotion is in >= n% of words => emotion exists
    return "{} {} {} {} {} {}".format(
        binaryResult(anger_count, limit),
        binaryResult(disgust_count, limit), 
        binaryResult(fear_count, limit), 
        binaryResult(joy_count, limit), 
        binaryResult(sadness_count, limit), 
        binaryResult(surprise_count, limit)
    )

#============================Load data for training and for testing============================
def processData(data, outputFilePathEmotions, outputFilePathValence):
    fileEmotions = open(outputFilePathEmotions, 'w+')
    fileValence = open(outputFilePathValence, 'w+')
    for id, phrase in data.items():
        phrase = preprocessPhrase(phrase)
        fileValence.write('{} {}\n'.format(id, processPhraseValence(phrase)))
        fileEmotions.write('{} {}\n'.format(id, processPhraseEmotions(phrase)))
    fileEmotions.close()
    fileValence.close()

def loadXmlData(xmlFilePath, data):
    xmldoc = minidom.parse(xmlFilePath)
    nodelist = xmldoc.getElementsByTagName('instance')
    for node in nodelist:
        data[node.getAttribute('id')] = node.firstChild.nodeValue.lower()

load_all_data()


loadXmlData('datasets/AffectiveText.trial/affectivetext_trial.xml', training_data)
loadXmlData('datasets/AffectiveText.test/affectivetext_test.xml', test_data)

processData(training_data, 'results/trial-emotions.gold', 'results/trial-valence.gold')
processData(test_data, 'results/test-emotions.gold', 'results/test-valence.gold')