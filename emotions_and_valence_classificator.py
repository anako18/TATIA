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

def load_datasets_sentiments():
    global negative_words
    global positive_words
    #read negative words
    f = open('datasets/sentiments/negative-words.txt', 'r') 
    negative_words = [word for line in f for word in line.split()]
    f.close()

    #read positive words
    f = open('datasets/sentiments/positive-words.txt', 'r') 
    positive_words = [word for line in f for word in line.split()]
    f.close()

#============================Process phrases============================
def binaryResult(value, limit):
    if value < limit:
        return "0"
    else:
        return "1"

#return -1 if negative 0 if neutral and 1 if positive
def processPhraseValence(phrase):
    negative_count = 0
    positive_count = 0
    total_count = 0
    phrase = phrase.replace('\n', '')
    phrase.translate(str.maketrans('', '', string.punctuation)) #remove punctuation signs
    words = nltk.word_tokenize(phrase)
    for word in words:
        word_lemmatized = lemmatizer.lemmatize(word)
        if word_lemmatized in positive_words:
            positive_count+=1
        elif word_lemmatized in negative_words:
            negative_count+=1
        total_count+=1
    if negative_count > positive_count:
        return "-1"
    elif negative_count < positive_count:
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
    phrase = phrase.replace('\n', '')
    phrase.translate(str.maketrans('', '', string.punctuation)) #remove punctuation signs
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

    limit = total_count*0.05 #if emotion is in >= 5% of words => emotion exists 
    return "{} {} {} {} {} {}".format(
        binaryResult(anger_count, limit),
        binaryResult(disgust_count, limit), 
        binaryResult(fear_count, limit), 
        binaryResult(joy_count, limit), 
        binaryResult(sadness_count, limit), 
        binaryResult(surprise_count, limit)
    )

#============================Load data for training and for testing============================
def processDataValence(data, outputFilePath):
    f = open(outputFilePath, 'w+')
    for id, phrase in data.items():
        f.write('{} {}\n'.format(id, processPhraseValence(phrase)))
    f.close()

def processDataEmotions(data, outputFilePath):
    f = open(outputFilePath, 'w+')
    for id, phrase in data.items():
        f.write('{} {}\n'.format(id, processPhraseEmotions(phrase)))
    f.close()

def loadXmlData(xmlFilePath, data):
    xmldoc = minidom.parse(xmlFilePath)
    nodelist = xmldoc.getElementsByTagName('instance')
    for node in nodelist:
        data[node.getAttribute('id')] = node.firstChild.nodeValue

load_datasets_emotions()
load_datasets_sentiments()

loadXmlData('datasets/AffectiveText.trial/affectivetext_trial.xml', training_data)
loadXmlData('datasets/AffectiveText.test/affectivetext_test.xml', test_data)

processDataValence(training_data, 'results/trial-valence.gold')
processDataValence(test_data, 'results/test-valence.gold')
processDataEmotions(training_data, 'results/trial-emotions.gold')
processDataEmotions(test_data, 'results/test-emotions.gold')