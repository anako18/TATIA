
import nltk
from nltk.stem import WordNetLemmatizer
import string

negative_words = []
positive_words = []
positive_count = 0
negative_count = 0
total_count = 0

def load_datasets():
    global negative_words
    global positive_words
    #read stop words
    #f = open('datasets/sentiments/stopwords.txt', 'r') 
    #stop_words = f.readlines()
    #f.close()

    #read negative words
    f = open('datasets/sentiments/negative-words.txt', 'r') 
    negative_words = [word for line in f for word in line.split()]
    f.close()

    #read positive words
    f = open('datasets/sentiments/positive-words.txt', 'r') 
    positive_words = [word for line in f for word in line.split()]
    f.close()

def sentiment_analysys():
    global positive_count
    global negative_count
    global total_count
    lemmatizer = WordNetLemmatizer()

    #text = input("Enter the text to analyse:")
    #"I'm super excited to see you at this excellent danger!" 
    f = open('test_documents/test1_positive.txt')
    text = f.read().replace('\n', '')
    text.translate(str.maketrans('', '', string.punctuation)) #remove punctuation signs
    words = nltk.word_tokenize(text)
    for word in words:
        word_lemmatized = lemmatizer.lemmatize(word)
        if word_lemmatized in positive_words:
            positive_count+=1
        elif word_lemmatized in negative_words:
            negative_count+=1
        total_count+=1

def print_result(negative_count, positive_count, total_count):
    neg = round(negative_count/total_count*100, 2)
    pos = round(positive_count/total_count*100, 2)
    neut = 100 - neg - pos
    res = 'NEUTRAL'
    if neg > pos:
        res = 'NEGATIVE'
    elif pos > neg:
        res = 'POSITIVE'
    print("This text is {}% positive, {}% negative, {}% neutral.\nThe result sentiment is {}".format(pos, neg, neut, res))

load_datasets()
sentiment_analysys()
print_result(negative_count, positive_count, total_count)
