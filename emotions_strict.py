
import nltk
from nltk.stem import WordNetLemmatizer
import string

anger_words = []
disgust_words = []
fear_words = []
joy_words = []
sadness_words = []
surprise_words = []

anger_count = 0
disgust_count = 0
fear_count = 0
joy_count = 0
sadness_count = 0
surprise_count = 0
total_count = 0

def load_datasets():
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

def sentiment_analysys():
    global anger_count
    global disgust_count
    global fear_count
    global joy_count
    global sadness_count
    global surprise_count
    global total_count

    lemmatizer = WordNetLemmatizer()

    #text = input("Enter the text to analyse:")
    f = open('test_documents/christmas_sentiments.txt')
    text = f.read().replace('\n', '')
    text.translate(str.maketrans('', '', string.punctuation)) #remove punctuation signs
    words = nltk.word_tokenize(text)
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

def print_result():
    global anger_count
    global disgust_count
    global fear_count
    global joy_count
    global sadness_count
    global surprise_count
    global total_count
    anger = round(anger_count/total_count*100, 2)
    disgust = round(disgust_count/total_count*100, 2)
    fear = round(fear_count/total_count*100, 2)
    joy = round(joy_count/total_count*100, 2)
    sadness = round(sadness_count/total_count*100, 2)
    surprise = round(surprise_count/total_count*100, 2)
    neut = 100 - anger - disgust - fear - joy - sadness - surprise

    emotions = {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
        "surprise": surprise
    }

    res = max(emotions, key=emotions. get)
    if all(value == 0 for value in emotions.values()):
        res = "neutral"
    print("This text is {}% anger, {}% disgust, {}% fear, {}% joy, {}% sadness, {}% neutral.\nThe result sentiment is {}".format(anger, disgust, fear, joy, sadness, neut, res))

load_datasets()
sentiment_analysys()
print_result()
