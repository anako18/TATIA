import nltk
from nltk.stem import WordNetLemmatizer
import xml.dom.minidom
from xml.dom import minidom
import string
import csv

limit = 20

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
        else:
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
    results = dict()
    f = open(file, 'r')
    reader = csv.reader(f)
    f.close
    for row in reader:
        results.update({row[0] : int(row[1])})
    return results
        
def convert_csvEmotions_to_dict(file):
    results = dict()
    f = open(file, 'r')
    reader = csv.reader(f)
    f.close
    for row in reader:
        results.update({row[0] : { "anger" : int(row[1]) , "disgust" : int(row[2]) ,"fear" : int(row[3]), "joy": int(row[4]) ,"sadness": int(row[5]),"surprise": int(row[6])}})

    return results
    
def addValue(my_dic,word, exist):
    if my_dic.get(word) == None:
        my_dic.update({word : 0})  
                    
    if exist == 1:
        value = my_dic.get(word) +1
        my_dic.update({word : value})
        
def addValueEmotion(my_dic1,my_dic2,word, exist):
    if my_dic1.get(word) == None:
        my_dic1.update({word : 0})
    if my_dic2.get(word) == None:
        my_dic2.update({word : 0}) 

    if exist == 1:
        value = my_dic1.get(word) +1 
        my_dic1.update({word : value})
    else :
        value = my_dic2.get(word) +1 
        my_dic2.update({word : value})
        

occWords = dict() 
proba_anger = 0
proba_Notanger = 0

proba_word_anger = dict()
proba_word_Notanger = dict()

words_in_anger_phrase = dict()
words_in_Notanger_phrase = dict()

proba_disgust = 0
proba_Notdisgust = 0

proba_word_disgust = dict()
proba_word_Notdisgust = dict()

words_in_disgust_phrase = dict()
words_in_Notdisgust_phrase = dict()

proba_fear = 0
proba_Notfear = 0

proba_word_fear = dict()
proba_word_Notfear = dict()

words_in_fear_phrase = dict()
words_in_Notfear_phrase = dict()

proba_joy = 0
proba_Notjoy = 0

proba_word_joy = dict()
proba_word_Notjoy = dict()

words_in_joy_phrase = dict()
words_in_Notjoy_phrase = dict()

proba_sad = 0
proba_Notsad = 0

proba_word_sad = dict()
proba_word_Notsad = dict()

words_in_sad_phrase = dict()
words_in_Notsad_phrase = dict()



proba_surprise = 0
proba_Notsurprise = 0
proba_word_surprise = dict()
proba_word_Notsurprise = dict()


words_in_surprise_phrase = dict()
words_in_Notsurprise_phrase = dict()



def proba_word_emotion(words_in_emotion_phrase,proba_emotion, word):
    global occWords
    if words_in_emotion_phrase.get(word) != None:
        proba  = words_in_emotion_phrase.get(word) / occWords.get(word)
        proba_emotion.update({word : proba})  
    else :
        proba_emotion.update({word : 0.001})   # to avoid 0 

        
    

def trainingEmotion(data,results):
    global occWords
    global proba_anger
    global proba_Notanger
    
    global proba_word_anger
    global words_in_anger_phrase
    
    global proba_word_Notanger
    global words_in_Notanger_phrase
    
    global proba_disgust
    global proba_Notdisgust

    global proba_word_disgust
    global words_in_disgust_phrase
    
    global proba_word_Notdisgust
    global words_in_Notdisgust_phrase
    
    global proba_fear
    global proba_Notfear
    
    global proba_word_fear
    global words_in_fear_phrase
    
    global proba_word_Notfear
    global words_in_Notfear_phrase

    global proba_joy
    global proba_Notjoy
    
    global proba_word_joy
    global words_in_joy_phrase
    
    global proba_word_Notjoy
    global words_in_Notjoy_phrase


    global proba_sad
    global proba_Notsad
    
    global proba_word_sad
    global words_in_sad_phrase
    
    global proba_word_Notsad
    global words_in_Notsad_phrase
    
    global proba_surprise
    global proba_Notsurprise
    
    global proba_word_surprise
    global words_in_surprise_phrase
    
    global proba_word_Notsurprise
    global words_in_Notsurprise_phrase
    
    occ_anger = 0
    occ_disgust = 0
    occ_fear = 0
    occ_joy = 0
    occ_sad = 0
    occ_surprise = 0
    occ_Notanger = 0
    occ_Notdisgust = 0
    occ_Notfear = 0
    occ_Notjoy = 0
    occ_Notsad = 0
    occ_Notsurprise = 0
    
    for id, phrase in data.items():
        phrase = preprocessPhrase(phrase)
        words = nltk.word_tokenize(phrase)
        for word in words:
            word = lemmatizer.lemmatize(word)
            addValue(occWords,word,1)
            
            addValueEmotion(words_in_anger_phrase,words_in_Notanger_phrase,word,results.get(id).get("anger"))
            addValueEmotion(words_in_disgust_phrase,words_in_Notdisgust_phrase,word,results.get(id).get("disgust"))
            addValueEmotion(words_in_fear_phrase,words_in_Notfear_phrase,word,results.get(id).get("fear"))
            addValueEmotion(words_in_joy_phrase,words_in_Notjoy_phrase,word,results.get(id).get("joy"))
            addValueEmotion(words_in_sad_phrase,words_in_Notsad_phrase,word,results.get(id).get("sadness"))
            addValueEmotion(words_in_surprise_phrase,words_in_Notsurprise_phrase,word,results.get(id).get("surprise"))

                
        if results.get(id).get("anger") == 1:  
            occ_anger +=1
        else:
            occ_Notanger +=1
            
        if results.get(id).get("disgust") == 1:  
            occ_disgust +=1
        else:
            occ_Notdisgust +=1
            
        if results.get(id).get("fear") == 1:  
            occ_fear +=1
        else:
            occ_Notfear +=1
            
        if results.get(id).get("joy") == 1:  
            occ_joy +=1
        else:
            occ_Notjoy +=1
            
        if results.get(id).get("sadness") == 1:  
            occ_sad +=1
        else:
            occ_Notsad +=1
            
        if results.get(id).get("surprise") == 1:  
            occ_surprise +=1
        else:
            occ_Notsurprise +=1
            
    # proba word givin emotion P(word/emotion)
    for key in occWords:
        proba_word_emotion(words_in_anger_phrase,proba_word_anger,key)    
        proba_word_emotion(words_in_Notanger_phrase,proba_word_Notanger,key)    
        
        proba_word_emotion(words_in_disgust_phrase,proba_word_disgust,key)   
        proba_word_emotion(words_in_Notdisgust_phrase,proba_word_Notdisgust,key)   
        
        proba_word_emotion(words_in_fear_phrase,proba_word_fear,key)  
        
        proba_word_emotion(words_in_joy_phrase,proba_word_joy,key)    

        proba_word_emotion(words_in_sad_phrase,proba_word_sad,key)    

        proba_word_emotion(words_in_surprise_phrase,proba_word_surprise,key)     
        
        proba_word_emotion(words_in_Notfear_phrase,proba_word_Notfear,key)  
        
        proba_word_emotion(words_in_Notjoy_phrase,proba_word_Notjoy,key)    

        proba_word_emotion(words_in_Notsad_phrase,proba_word_Notsad,key)    

        proba_word_emotion(words_in_Notsurprise_phrase,proba_word_Notsurprise,key)    
        
    ### probabilie des emotions P(emotion)
    proba_anger  = occ_anger / len(data.items())        
    proba_disgust  = occ_disgust / len(data.items())        
    proba_fear  = occ_fear / len(data.items())        
    proba_joy  = occ_joy / len(data.items())        
    proba_sad  = occ_sad / len(data.items())        
    proba_surprise  = occ_surprise / len(data.items())     
    proba_Notanger  = occ_Notanger / len(data.items())        
    proba_Notdisgust  = occ_Notdisgust / len(data.items())        
    proba_Notfear  = occ_Notfear / len(data.items())        
    proba_Notjoy  = occ_Notjoy / len(data.items())        
    proba_Notsad  = occ_Notsad / len(data.items())        
    proba_Notsurprise  = occ_Notsurprise / len(data.items())   
    

def calculateProbaBayes(words, dic , probabity):
    proba = probabity
    for word in words:
        word = lemmatizer.lemmatize(word)
        value = dic.get(word) 
        if value == None or value == 0:
            value = 0.1
        proba = proba * value
        
    # device by some thing
    return proba
    
def processPhraseEmotions(phrase):
    global occWords
    global proba_anger
    global proba_Notanger
    global proba_word_anger
    global proba_word_Notanger
    
    global proba_disgust
    global proba_Notdisgust
    global proba_word_disgust
    global proba_word_Notdisgust
    
    global proba_fear
    global proba_Notfear   
    global proba_word_fear   
    global proba_word_Notfear

    global proba_joy
    global proba_Notjoy  
    global proba_word_joy   
    global proba_word_Notjoy

    global proba_sad
    global proba_Notsad   
    global proba_word_sad    
    global proba_word_Notsad
    
    global proba_surprise
    global proba_Notsurprise    
    global proba_word_surprise   
    global proba_word_Notsurprise
    
    words = nltk.word_tokenize(phrase)
    
    angerproba = calculateProbaBayes(words,proba_word_anger, proba_anger )
    disgustproba = calculateProbaBayes(words,proba_word_disgust, proba_disgust )
    fearproba = calculateProbaBayes(words,proba_word_fear, proba_fear )
    joyproba = calculateProbaBayes(words,proba_word_joy, proba_joy )
    sadproba = calculateProbaBayes(words,proba_word_sad, proba_sad )
    surpriseproba = calculateProbaBayes(words,proba_word_surprise, proba_surprise)
    
    notangerproba = calculateProbaBayes(words,proba_word_Notanger, proba_Notanger )
    notdisgustproba = calculateProbaBayes(words,proba_word_Notdisgust, proba_Notdisgust )
    notfearproba = calculateProbaBayes(words,proba_word_Notfear, proba_Notfear )
    notjoyproba = calculateProbaBayes(words,proba_word_Notjoy, proba_Notjoy )
    notsadproba = calculateProbaBayes(words,proba_word_Notsad, proba_Notsad )
    notsurpriseproba = calculateProbaBayes(words,proba_word_Notsurprise, proba_Notsurprise)    
    
    
    
    
    return "{} {} {} {} {} {}".format(
        binaryResult(angerproba, notangerproba),
        binaryResult(disgustproba, notdisgustproba), 
        binaryResult(fearproba, notfearproba), 
        binaryResult(joyproba, notjoyproba), 
        binaryResult(sadproba, notsadproba), 
        binaryResult(surpriseproba, notsurpriseproba)
    )

   



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

def trainingValence(data,results):
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
    occ_positive = 0
    occ_negative = 0
    occ_neutral = 0
    for id, phrase in data.items():
        phrase = preprocessPhrase(phrase)
        words = nltk.word_tokenize(phrase)
        polarity_phrase = results.get(str(id))
                
        if polarity_phrase == 1:
            occ_positive += 1
        if polarity_phrase == -1:
            occ_negative += 1
        if polarity_phrase == 0:
            occ_neutral += 1
            
        for word in words:
            word = lemmatizer.lemmatize(word)
            addValue(occWords,word,1)
                          
            if polarity_phrase == 1:
                addValue(words_in_positive_phrase,word,1)

            if polarity_phrase == -1:
                addValue(words_in_negative_phrase,word,1)

            if polarity_phrase == 0:
                addValue(words_in_neutral_phrase,word,1)

        for key in occWords:
            proba_word_emotion(words_in_positive_phrase,proba_positive_word,key)
            proba_word_emotion(words_in_negative_phrase,proba_negative_word,key)
            proba_word_emotion(words_in_neutral_phrase,proba_neutral_word,key)

    
    proba_positive = occ_positive  / len(data.items()) 
    proba_negative = occ_negative / len(data.items())      
    proba_neutral  = occ_neutral / len(data.items())  




def processPhraseValence(phrase):
    global occWords
    global proba_positive_word 
    global proba_negative_word 
    global proba_neutral_word
    global proba_negative
    global proba_positive 
    global proba_neutral
    words = nltk.word_tokenize(phrase)

    
    positiveproba = calculateProbaBayes(words,proba_positive_word, proba_positive )
    negativeproba = calculateProbaBayes(words,proba_negative_word, proba_negative )
    neutralproba = calculateProbaBayes(words,proba_neutral_word, proba_neutral )
    
    positiveproba = positiveproba /(positiveproba+negativeproba+neutralproba)
    negativeproba = negativeproba /(positiveproba+negativeproba+neutralproba)
    neutralproba = neutralproba /(positiveproba+negativeproba+neutralproba)
    
     

    if positiveproba > neutralproba + negativeproba:
        return "1"
    if  negativeproba > neutralproba + positiveproba  : # neutralproba + positiveproba = notnegativeproba 
        return "-1"
    return "0"
        

    

def processTest(data, outputFilePathValence,outputFilePathEmotion):
    fileValence = open(outputFilePathValence, 'w+')
    fileEmotion = open(outputFilePathEmotion, 'w+')
    for id, phrase in data.items():
        phrase = preprocessPhrase(phrase)
        fileValence.write('{} {}\n'.format(id, processPhraseValence(phrase)))
        fileEmotion.write('{} {}\n'.format(id, processPhraseEmotions(phrase)))

def training(training_data,FileEmotion,FileValence):
    valenceDict = convert_csvValence_to_dict(FileValence)
    emotionDict = convert_csvEmotions_to_dict(FileEmotion)
    trainingValence(training_data,valenceDict)   
    trainingEmotion(training_data,emotionDict)


        
training_data = {}
test_data ={}

loadXmlData('datasets/AffectiveText.trial/affectivetext_trial.xml', training_data)

training(training_data,"datasets/AffectiveText.trial/emotion.csv","datasets/AffectiveText.trial/valence.csv")
 

loadXmlData('datasets/AffectiveText.test/affectivetext_test.xml', test_data)

outputfileTestEmotion = "results/test_emotions_Naive_Bayes.gold"

outputfileTestValence = "results/test_valence_Naive_Bayes.gold"

processTest(test_data,outputfileTestValence,outputfileTestEmotion)        
