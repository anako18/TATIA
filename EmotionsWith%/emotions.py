import csv
import nltk
from nltk.stem import WordNetLemmatizer

wordList =[]
disgustList =[]
surpriseList =[]
neutralList =[]
angerList =[]
sadList =[]
happyList =[]
fearList =[]
wordnet_lemmatizer = WordNetLemmatizer()

# create list 
with open('Andbrain_DataSet.csv', 'r') as f:
    reader = csv.reader(f , delimiter=',')
    for row in reader:
            wordList.append(wordnet_lemmatizer.lemmatize(row[0].strip()))
            disgustList.append(row[1].strip())
            surpriseList.append(row[2].strip())
            neutralList.append(row[3].strip())
            angerList.append(row[4].strip())
            sadList.append(row[5].strip())
            happyList.append(row[6].strip())
            fearList.append(row[7].strip())
            

# index of elment in word list 
index = -1 

disgust = 0
surprise = 0
neutral = 0
anger = 0
sad = 0
happy = 0
fear = 0

punctuations="?:!.,;"

sentence ='I like the cat of my neighbor!'
sentence_words = nltk.word_tokenize(sentence)
for word in sentence_words:
    if word in punctuations:
        sentence_words.remove(word)
    else :
        if wordnet_lemmatizer.lemmatize(word) in wordList:
            index = wordList.index(word)
            disgust += float(disgustList[index])
            surprise += float(surpriseList[index])
            neutral += float(neutralList[index])
            anger += float(angerList[index])
            sad += float(sadList[index])
            happy += float(happyList[index])
            fear += float(fearList[index])
        

emotions = {"disgust":disgust,"surprise":surprise,"neutral":neutral,"anger":anger,"sad":sad,"happy":happy,"fear":fear}
# give the emotion 
MaxKey = max(emotions, key=emotions.get)
if emotions[MaxKey] == 0:
    MaxKey = 'Don\'t know'
else :
    print({key: value for key, value in sorted(emotions.items(), key=lambda item: item[1])})

print("the emotion of the sentence is :",MaxKey)
print(MaxKey)
