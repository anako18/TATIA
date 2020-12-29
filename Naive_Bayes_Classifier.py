import nltk
from nltk.stem import WordNetLemmatizer
import xml.dom.minidom
from xml.dom import minidom
import string
import csv

"""
rename functions and variables
pre proccess phrase add to dic lematize word 
"""

limit = 10

def preprocessPhrase(phrase):
    phrase = phrase.replace('\n', '')
    phrase = phrase.translate(str.maketrans('', '', string.punctuation)) #remove punctuation signs
    return phrase
    
def binaryResult(value, limit):
    if value < limit:
        return "0"
    else:
        return "1"

lemmatizer = WordNetLemmatizer()

def loadXmlData(xmlFilePath, data):
    xmldoc = minidom.parse(xmlFilePath)
    nodelist = xmldoc.getElementsByTagName('instance')
    for node in nodelist:
        data[node.getAttribute('id')] = node.firstChild.nodeValue.lower()
        


def convertValenceFile(file1, file2):
    result = []
    lenth = 0
    f = open(file1, 'r') 
    result = [int(word) for line in f for word in line.split()]
    f.close()
    file = open(file2, 'w')
    i = 0
    while i < len(result):
        value = result[i+1]
        index = result[i]
        if value > 0:
            file.write(str(index) +", 1\n")
        if value < 0:
            file.write(str(index) +", -1\n")
        if value == 0:
            file.write(str(index) +", 0\n")
        lenth +=1
        i+=2
    file.close()
    
def convertEmotionFile(file1,file2):
    global lines
    global limit 
    f = open(file1, 'r') 
    lines = [line for line in f]
    f.close()
    f = open(file2, 'w') 
    for line in lines:
        result = [int(l) for l in line.split() ]
        i = 1
        f.write( str(result[0]))
        while i < len(result):
           f.write( ","+ binaryResult(result[i],limit) )
           i+=1
        f.write("\n")  
        
    f.close()    
    
    
results =  dict()

def convert_csvValence_to_dict(file):
    global results
    f = open(file, 'r')
    reader = csv.reader(f)
    f.close
    for row in reader:
        
        results.update({row[0] : int(row[1])})
        
def convert_csvEmotions_to_dict(file):
    global results
    f = open(file, 'r')
    reader = csv.reader(f)
    f.close
    for row in reader:
        
        results.update({row[0] : { "anger" : int(row[1]) , "disgust" : int(row[2]) ,"fear" : int(row[3]), "joy": int(row[4]) ,"sadness": int(row[5]),"surprise": int(row[6])}})



#print(results.get("1").get("disgust"))
occWords = dict() 
proba_anger = 0
proba_anger_word = dict()
words_in_anger_phrase = dict()

proba_disgust = 0
proba_disgust_word = dict()
words_in_disgust_phrase = dict()

proba_fear = 0
proba_fear_word = dict()
words_in_fear_phrase = dict()

proba_joy = 0
proba_joy_word = dict()
words_in_joy_phrase = dict()

proba_sad = 0
proba_sad_word = dict()
words_in_sad_phrase = dict()

proba_surprise = 0
proba_surprise_word = dict()
words_in_surprise_phrase = dict()


def addValue(my_dic,word, exist):
    if my_dic.get(word) == None:
        my_dic.update({word : 0})  
                    
    if exist == 1:
        value = my_dic.get(word) +1
        my_dic.update({word : value})

    

def trainingEmotion(data):
    global occWords
    global proba_anger
    global proba_anger_word
    global words_in_anger_phrase

    global proba_disgust
    global proba_disgust_word
    global words_in_disgust_phrase

    global proba_fear
    global proba_fear_word
    global words_in_fear_phrase

    global proba_joy
    global proba_joy_word
    global words_in_joy_phrase

    global proba_sad
    global proba_sad_word
    global words_in_sad_phrase

    global proba_surprise
    global proba_surprise_word
    global words_in_surprise_phrase
    
    result_anger = 0
    result_disgust = 0
    result_fear = 0
    result_joy = 0
    result_sad = 0
    result_surprise = 0
    for id, phrase in data.items():
        phrase = preprocessPhrase(phrase)
        words = nltk.word_tokenize(phrase)
        for word in words:
            word = lemmatizer.lemmatize(word)
            value = occWords.get(word)
            if value != None:
                value +=1
            else:
                value = 1
            occWords.update({word : value})
            addValue(words_in_anger_phrase,word,results.get(id).get("anger"))
            addValue(words_in_disgust_phrase,word,results.get(id).get("disgust"))
            addValue(words_in_fear_phrase,word,results.get(id).get("fear"))
            addValue(words_in_joy_phrase,word,results.get(id).get("joy"))
            addValue(words_in_sad_phrase,word,results.get(id).get("sadness"))
            addValue(words_in_surprise_phrase,word,results.get(id).get("surprise"))

                
        if results.get(id).get("anger") == 1:  
            result_anger +=1
        if results.get(id).get("disgust") == 1:  
            result_disgust +=1
        if results.get(id).get("fear") == 1:  
            result_fear +=1
        if results.get(id).get("joy") == 1:  
            result_joy +=1
        if results.get(id).get("sadness") == 1:  
            result_sad +=1
        if results.get(id).get("surprise") == 1:  
            result_surprise +=1
        
    for key in occWords:
        proba  = words_in_anger_phrase.get(key) / occWords.get(key)
        proba_anger_word.update({key : proba})  
        
        proba  = words_in_disgust_phrase.get(key) / occWords.get(key)
        proba_disgust_word.update({key : proba})  
        
        proba  = words_in_fear_phrase.get(key) / occWords.get(key)
        proba_fear_word.update({key : proba})   
        
        proba  = words_in_joy_phrase.get(key) / occWords.get(key)
        proba_joy_word.update({key : proba})   
        
        proba  = words_in_sad_phrase.get(key) / occWords.get(key)
        proba_sad_word.update({key : proba})   
        
        proba  = words_in_surprise_phrase.get(key) / occWords.get(key)
        proba_surprise_word.update({key : proba})    
        
    proba_anger  = result_anger / len(data.items())        
    proba_disgust  = result_disgust / len(data.items())        
    proba_fear  = result_fear / len(data.items())        
    proba_joy  = result_joy / len(data.items())        
    proba_sad  = result_sad / len(data.items())        
    proba_surprise  = result_surprise / len(data.items())        

def calculateProba(words, dic , probabity):
    proba = probabity
    for word in words:
        word = lemmatizer.lemmatize(word)
        value = dic.get(word) 
        if value == None or value == 0:
            value = 0.001
        proba = proba * value
        
    # device by some thing
    return proba
    
def processPhraseEmotions(phrase):
    global occWords
    global proba_anger
    global proba_anger_word
    
    global proba_disgust
    global proba_disgust_word

    global proba_fear
    global proba_fear_word

    global proba_joy
    global proba_joy_word

    global proba_sad
    global proba_sad_word

    global proba_surprise
    global proba_surprise_word
    
    words = nltk.word_tokenize(phrase)
    
    angerproba = calculateProba(words,proba_anger_word, proba_anger )
    disgustproba = calculateProba(words,proba_disgust_word, proba_disgust )
    fearproba = calculateProba(words,proba_fear_word, proba_fear )
    joyproba = calculateProba(words,proba_joy_word, proba_joy )
    sadproba = calculateProba(words,proba_sad_word, proba_sad )
    surpriseproba = calculateProba(words,proba_surprise_word, proba_surprise)
    
    angerproba = angerproba /(angerproba+disgustproba+fearproba+joyproba+sadproba +surpriseproba)
    disgustproba = disgustproba /(angerproba+disgustproba+fearproba+joyproba+sadproba+surpriseproba)
    fearproba = fearproba /(angerproba+disgustproba+fearproba+joyproba+sadproba+surpriseproba)
    joyproba = joyproba /(angerproba+disgustproba+fearproba+joyproba+sadproba+surpriseproba)
    sadproba = sadproba /(angerproba+disgustproba+fearproba+joyproba+sadproba+surpriseproba)
    surpriseproba = surpriseproba /(angerproba+disgustproba+fearproba+joyproba+sadproba+surpriseproba)
    
    limit = (max(angerproba,disgustproba,fearproba,joyproba,sadproba ,surpriseproba)/3)
    return "{} {} {} {} {} {}".format(
        binaryResult(angerproba, limit),
        binaryResult(disgustproba, limit), 
        binaryResult(fearproba, limit), 
        binaryResult(joyproba, limit), 
        binaryResult(sadproba, limit), 
        binaryResult(surpriseproba, limit)
    )

   


results = dict()

occWords = dict() 
words_in_positive_phrase = dict()  
words_in_negative_phrase = dict()   
words_in_neutral_phrase = dict()   
proba_positive_word = dict()
proba_negative_word = dict()
proba_neutral_word  = dict()

proba_negative = 0
proba_positive = 0
proba_neutral  = 0

def training(data):
    global results
    global occWords
    global words_in_positive_phrase 
    global words_in_negative_phrase 
    global words_in_neutral_phrase
    global proba_negative
    global proba_positive 
    global proba_neutral
    global proba_positive_word 
    global proba_negative_word 
    global proba_neutral_word
    occWords = dict()
    result_positive = 0
    result_negative = 0
    result_neutral = 0
    for id, phrase in data.items():
        phrase = preprocessPhrase(phrase)
        words = nltk.word_tokenize(phrase)
        polarity_phrase = results.get(str(id))
                
        if polarity_phrase == 1:
            result_positive += 1
        if polarity_phrase == -1:
            result_negative += 1
        if polarity_phrase == 0:

            result_neutral += 1
        #print(str(id) +" "+ str(polarity_phrase))
        for word in words:
            word = lemmatizer.lemmatize(word)
            value = occWords.get(word)
            if value != None:
                value +=1
            else:
                value = 1
            occWords.update({word : value})

            if words_in_positive_phrase.get(word) == None:
                words_in_positive_phrase.update({word : 0})
            if words_in_negative_phrase.get(word) == None:
                words_in_negative_phrase.update({word : 0})
            if words_in_neutral_phrase.get(word) == None:
                words_in_neutral_phrase.update({word : 0})
                
            if polarity_phrase == 1:
                value = words_in_positive_phrase.get(word) +1
                words_in_positive_phrase.update({word : value})
            if polarity_phrase == -1:
                value = words_in_negative_phrase.get(word)+1
                words_in_negative_phrase.update({word : value})
            if polarity_phrase == 0:
                value = words_in_neutral_phrase.get(word)+1
                words_in_neutral_phrase.update({word : value})
                       
        
        for key in occWords:
            proba_word  = words_in_positive_phrase.get(key) / occWords.get(key)
            proba_positive_word.update({key : proba_word})
            
            proba_word  = words_in_negative_phrase.get(key) / occWords.get(key)
            proba_negative_word.update({key : proba_word})
            
            proba_word  = words_in_neutral_phrase.get(key) / occWords.get(key)
            proba_neutral_word.update({key : proba_word})     
            #print(key  +" pos "+ str(words_in_positive_phrase.get(key))+ " " +str(proba_positive_word.get(key)) +" neg "+ str(words_in_negative_phrase.get(key)) +" "+ str(proba_negative_word.get(key))+" / "+ str(occWords.get(key))  )
    
    proba_positive = result_positive  / len(data.items()) 
    proba_negative = result_negative / len(data.items())      
    proba_neutral  = result_neutral / len(data.items())  




def processPhraseValence(phrase):
    global occWords
    global proba_positive_word 
    global proba_negative_word 
    global proba_neutral_word
    global proba_negative
    global proba_positive 
    global proba_neutral
    words = nltk.word_tokenize(phrase)

    
    positiveproba = calculateProba(words,proba_positive_word, proba_positive )
    negativeproba = calculateProba(words,proba_negative_word, proba_negative )
    neutralproba = calculateProba(words,proba_neutral_word, proba_neutral )
    
    positiveproba = positiveproba /(positiveproba+negativeproba+neutralproba)
    negativeproba = negativeproba /(positiveproba+negativeproba+neutralproba)
    neutralproba = neutralproba /(positiveproba+negativeproba+neutralproba)
    
    #maximun(positiveproba, positiveproba,neutralproba)
     
    if  negativeproba > neutralproba and  negativeproba > positiveproba :
        return "-1"
    if positiveproba > neutralproba and  positiveproba > negativeproba:
        return "1"

    return "0"
        

    

def processTest(data, outputFilePathValence,outputFilePathEmotion):
    fileValence = open(outputFilePathValence, 'w+')
    fileEmotion = open(outputFilePathEmotion, 'w+')
    for id, phrase in data.items():
        phrase = preprocessPhrase(phrase)
        fileValence.write('{} {}\n'.format(id, processPhraseValence(phrase)))
        fileEmotion.write('{} {}\n'.format(id, processPhraseEmotions(phrase)))


        
training_data = {}
test_data ={}

loadXmlData('datasets/AffectiveText.trial/affectivetext_trial.xml', training_data)


fTrial = "datasets/AffectiveText.trial/affectivetext_trial.emotions.gold"
fileTrial = "datasets/AffectiveText.trial/emotion.csv"
        
convertEmotionFile(fTrial,fileTrial)   
convert_csvEmotions_to_dict(fileTrial)
trainingEmotion(training_data)

fTrial = "datasets/AffectiveText.trial/affectivetext_trial.valence.gold"
fileTrial = "datasets/AffectiveText.trial/valence.csv"
        
convertValenceFile(fTrial,fileTrial)   
convert_csvValence_to_dict(fileTrial)
training(training_data)   

loadXmlData('datasets/AffectiveText.test/affectivetext_test.xml', test_data)

outputfileTestEmotion = "datasets/AffectiveText.test/Naive_Bayes_emotions.gold"

outputfileTestValence = "datasets/AffectiveText.test/Naive_Bayes_valence.gold"


processTest(test_data,outputfileTestValence,outputfileTestEmotion)        
