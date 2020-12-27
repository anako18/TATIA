import nltk
from nltk.stem import WordNetLemmatizer
import xml.dom.minidom
from xml.dom import minidom
import string
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 



import spacy
from spacy.tokenizer import Tokenizer
nlp = spacy.load('en_core_web_sm')
tokenizer = Tokenizer(nlp.vocab)

s = "something good\nSomething is an and love 2 somethings"
tokens = tokenizer(s)
"""
for token in tokens:
  print(token.lemma_, token.is_stop)
"""

disgust_words = []
fear_words = []
joy_words = []
sadness_words = []
surprise_words = []
anger_words =[]
#valence words
negative_words = []
positive_words = []

stop_words = []

lemmatizer = WordNetLemmatizer()

#============================Load datasets============================
def load_datasets_emotions():
    global anger_words
    global disgust_words
    global fear_words
    global joy_words
    global sadness_words
    global surprise_words
    global stop_words

     #read stop words
    f = open('datasets/sentiments/stopwordsSynonyms.txt', 'r') 
    stop_words = [word.lower() for line in f for word in line.split()]
    f.close()

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
    f = open('datasets/sentiments/negativewordsSynonyms.txt', 'r') 
    negative_words = [word.lower() for line in f for word in line.split()]
    f.close()

    #read positive words
    f = open('datasets/sentiments/positivewordsSynonyms.txt', 'r') 
    positive_words = [word.lower() for line in f for word in line.split()]
    f.close()
    
    





load_datasets_emotions()
load_datasets_sentiments()




spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

existings_words = anger_words + disgust_words + fear_words + joy_words + sadness_words + surprise_words + stop_words +negative_words + positive_words

negativeWords = anger_words+ disgust_words +sadness_words +negative_words+fear_words
positiveWords = positive_words + joy_words 

positiveFile = open("positive.txt", "a")
negativeFile = open("negative.txt", "a")
nonFile = open("nonFile.txt", "a")

for word in negativeWords:
  if word not in spacy_stopwords:
      feedback_polarity = TextBlob(lemmatizer.lemmatize(word)).sentiment.polarity
      if feedback_polarity>0:
        positiveWords.append(word)
        negativeWords.remove(word)
      if feedback_polarity ==0:
        negativeWords.remove(word)
  else:
    negativeWords.remove(word)

        
for word in positiveWords:
  if word not in spacy_stopwords:
      feedback_polarity = TextBlob(lemmatizer.lemmatize(word)).sentiment.polarity
      if feedback_polarity<0:
        negativeWords.append(word)
        positiveWords.remove(word) 
      if feedback_polarity ==0:
        positiveWords.remove(word)
  else:
      positiveWords.remove(word)
    
       


sid_obj = SentimentIntensityAnalyzer() 

    
for word in existings_words:
  if word not in spacy_stopwords and  word not in positiveWords and word not in negativeWords:
      feedback_polarity = TextBlob(lemmatizer.lemmatize(word)).sentiment.polarity 
      if feedback_polarity < 0:
        negativeWords.append(word)
      if feedback_polarity > 0:
        positiveWords.append(word)
      if feedback_polarity == 0:
        sentiment_dict = sid_obj.polarity_scores(word) 
        if sentiment_dict['compound'] >= 0.05 : 
            positiveWords.append(word)
        if sentiment_dict['compound'] <= - 0.05 : 
            negativeWords.append(word)
        else : 
            nonFile.write(word+"\n") 
            
for word in positiveWords:
    if word in negativeWords:
        positiveWords.remove(word)
        negativeWords.remove(word)
        
        
for word in negativeWords:
    if word in positiveWords:
        positiveWords.remove(word)
        negativeWords.remove(word)
   
for word in negativeWords:
    negativeFile.write(word+"\n")
    
for word in positiveWords:
    positiveFile.write(word+"\n")
positiveFile.close()
negativeFile.close()
nonFile.close()



"""
for stop_word in spacy_stopwords:   
    neutralFile.write(stop_word)
    neutralFile.write("\n")
"""

"""       
sid_obj = SentimentIntensityAnalyzer() 
sentiment_dict = sid_obj.polarity_scores(word) 
if sentiment_dict['compound'] >= 0.05 : 
if sentiment_dict['compound'] <= - 0.05 : 
    print("Negative") 
else : 
    print("Neutral") 


for word in positiveWords:
  if word not in spacy_stopwords:
    sentiment_dict = sid_obj.polarity_scores(word) 
    if sentiment_dict['compound'] <= - 0.05 : 
        negativeWords.append(word)
    else : 
        print("Neutral") 
    
"""