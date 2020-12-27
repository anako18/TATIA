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
def load_datasets_emotions():
    global anger_words
    global disgust_words
    global fear_words
    global joy_words
    global sadness_words
    global surprise_words

    #read anger words
    f = open('datasets/WordNetAffectEmotionLists/angerSynonyms.txt', 'r') 
    anger_words = [word.lower() for line in f for word in line.split()]
    f.close()

    #read disgust words
    f = open('datasets/WordNetAffectEmotionLists/disgustSynonyms.txt', 'r') 
    disgust_words = [word.lower() for line in f for word in line.split()]
    f.close()

    #read fear words
    f = open('datasets/WordNetAffectEmotionLists/fearSynonyms.txt', 'r') 
    fear_words = [word.lower() for line in f for word in line.split()]
    f.close()

    #read joy words
    f = open('datasets/WordNetAffectEmotionLists/joySynonyms.txt', 'r') 
    joy_words = [word.lower() for line in f for word in line.split()]
    f.close()

    #read sadness words
    f = open('datasets/WordNetAffectEmotionLists/sadnessSynonyms.txt', 'r') 
    sadness_words = [word.lower() for line in f for word in line.split()]
    f.close()

    #read surprise words
    f = open('datasets/WordNetAffectEmotionLists/surpriseSynonyms.txt', 'r') 
    surprise_words = [word.lower() for line in f for word in line.split()]
    f.close()

def load_datasets_sentiments():
    global negative_words
    global positive_words
    #read negative words
    f = open('words/negative.txt', 'r') 
    negative_words = [word.lower() for line in f for word in line.split()]
    f.close()

    #read positive words
    f = open('words/positive.txt', 'r') 
    positive_words = [word.lower() for line in f for word in line.split()]
    f.close()

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
    total_count = 0
    words = nltk.word_tokenize(phrase)
    for word in words:
        word_lemmatized = lemmatizer.lemmatize(word)
        if word_lemmatized in positive_words:
            positive_count+=1
        elif word_lemmatized in negative_words:
            negative_count+=1
        total_count+=1
    limit = total_count*0.1
    if negative_count - positive_count > limit:
        return "-1"
    elif  positive_count - negative_count > limit:
        return "1"
    else:
        return "0"
        
        


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
        if word_lemmatized in anger_words:
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

load_datasets_emotions()
load_datasets_sentiments()

loadXmlData('datasets/AffectiveText.trial/affectivetext_trial.xml', training_data)
loadXmlData('datasets/AffectiveText.test/affectivetext_test.xml', test_data)

processData(training_data, 'results/trial-emotions.gold', 'resultswithwords/trial-valence_positive.gold')
processData(test_data, 'results/test-emotions.gold', 'resultswithwords/test-valence_positive.gold')
"""
processData(training_data, 'results/trial-emotions.gold', 'results/trial-valence.gold')
processData(test_data, 'results/test-emotions.gold', 'results/test-valence.gold')
"""
