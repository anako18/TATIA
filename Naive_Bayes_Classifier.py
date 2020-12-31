import nltk
from nltk.stem import WordNetLemmatizer
import xml.dom.minidom
from xml.dom import minidom
import string
import csv



threshold_emotions = 20
threshold_valence = 30
def preprocessPhrase(phrase):
    phrase = phrase.replace('\n', '')
    phrase = phrase.translate(str.maketrans('', '', string.punctuation)) #remove punctuation signs
    return phrase
    
def binaryResult(value, limit):
    if value < limit:
        return "0"
    else:
        return "1"
        
def tripleResult(value, threshold):
    if (value >= threshold):
        return "1"
    elif (value < threshold) and (value > -1*threshold):
        return "0"
    else:
        return "-1"

lemmatizer = WordNetLemmatizer()

def loadXmlData(xmlFilePath, data):
    xmldoc = minidom.parse(xmlFilePath)
    nodelist = xmldoc.getElementsByTagName('instance')
    for node in nodelist:
        data[node.getAttribute('id')] = node.firstChild.nodeValue.lower()
        


def convertValenceFiles(file1,file2, file3):
    result = []
    lenth = 0
    global threshold_valence
    f = open(file1, 'r') 
    result = [int(word) for line in f for word in line.split()]
    f.close()
    fileBinary = open(file2, 'w')
    fileTriple = open(file3, 'w')
    i = 0
    while i < len(result):
        value = result[i+1]
        index = result[i]
        fileBinary.write(str(index) +","+ binaryResult(value,threshold_valence)+"\n")
        fileTriple.write(str(index)+"," +tripleResult(value, threshold_valence)+"\n")
        i+=2
    fileBinary.close()
    fileTriple.close()
    
   
def convertEmotionFile(file1,file2):
    global lines
    global  threshold_emotions
    f = open(file1, 'r') 
    lines = [line for line in f]
    f.close()
    f = open(file2, 'w') 
    for line in lines:
        result = [int(l) for l in line.split() ]
        i = 1
        f.write( str(result[0]))
        while i < len(result):
           f.write( ","+ binaryResult(result[i],threshold_emotions) )
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
    occWords = dict()
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

words_in_positive_phraseBinary = dict()  
words_in_positive_phraseTriple = dict()  
words_in_negative_phraseBinary = dict()   
words_in_negative_phraseTriple = dict()   
words_in_neutral_phrase = dict()   
proba_positive_wordBinary = dict()
proba_positive_wordTriple = dict()
proba_negative_wordBinary = dict()
proba_negative_wordTriple = dict()
proba_neutral_word  = dict()

proba_negativeBinary = 0
proba_negativeTriple = 0
proba_positiveBinary = 0
proba_positiveTriple = 0
proba_neutral  = 0

def trainingTripleValence(data,results):
    global occWords 
    global words_in_positive_phraseTriple
    global words_in_negative_phraseTriple 
    global words_in_neutral_phrase
    global proba_negativeTriple 
    global proba_positiveTriple
    global proba_neutral 
    global proba_positive_wordTriple 
    global proba_negative_wordTriple 
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
                addValue(words_in_positive_phraseTriple,word,1)

            if polarity_phrase == -1:
                addValue(words_in_negative_phraseTriple,word,1)

            if polarity_phrase == 0:
                addValue(words_in_neutral_phrase,word,1)

        for key in occWords:
            proba_word_emotion(words_in_positive_phraseTriple,proba_positive_wordTriple,key)
            proba_word_emotion(words_in_negative_phraseTriple,proba_negative_wordTriple,key)
            proba_word_emotion(words_in_neutral_phrase,proba_neutral_word,key)

    
    proba_positiveTriple = occ_positive  / len(data.items()) 
    proba_negativeTriple = occ_negative / len(data.items())      
    proba_neutral  = occ_neutral / len(data.items())  

def trainingBinaryValence(data,results):
    global occWords 
    global words_in_positive_phraseBinary 
    global words_in_negative_phraseBinary 
    global proba_negativeBinary 
    global proba_positiveBinary 
    global proba_positive_wordBinary 
    global proba_negative_wordBinary 
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
        if polarity_phrase == 0:
            occ_negative += 1

            
        for word in words:
            word = lemmatizer.lemmatize(word)
            addValue(occWords,word,1)
                          
            if polarity_phrase == 1:
                addValue(words_in_positive_phraseBinary,word,1)

            if polarity_phrase == 0:
                addValue(words_in_negative_phraseBinary,word,1)

        for key in occWords:
            proba_word_emotion(words_in_positive_phraseBinary,proba_positive_wordBinary,key)
            proba_word_emotion(words_in_negative_phraseBinary,proba_negative_wordBinary,key)

    
    proba_positiveBinary = occ_positive  / len(data.items()) 
    proba_negativeBinary = occ_negative / len(data.items())      



def processPhraseValenceTriple(phrase):
    global occWords
    global proba_positive_wordTriple 
    global proba_negative_wordTriple
    global proba_neutral_word
    global proba_negativeTriple
    global proba_positiveTriple
    global proba_neutral
    words = nltk.word_tokenize(phrase)

    
    positiveproba = calculateProbaBayes(words,proba_positive_wordTriple, proba_positiveTriple )
    negativeproba = calculateProbaBayes(words,proba_negative_wordTriple, proba_negativeTriple )
    neutralproba = calculateProbaBayes(words,proba_neutral_word, proba_neutral )
    
    positiveproba = positiveproba /(positiveproba+negativeproba+neutralproba)
    negativeproba = negativeproba /(positiveproba+negativeproba+neutralproba)
    neutralproba = neutralproba /(positiveproba+negativeproba+neutralproba)
    
     

    if positiveproba > neutralproba +  negativeproba :
        return "1"
    if  negativeproba > neutralproba + positiveproba  : # neutralproba + positiveproba = notnegativeproba 
        return "-1"
    return "0"
        
def processPhraseValenceBinary(phrase):
    global occWords
    global proba_positive_wordBinary 
    global proba_negative_wordBinary 
    global proba_negativeBinary
    global proba_positiveBinary
    words = nltk.word_tokenize(phrase)
    
    positiveproba = calculateProbaBayes(words,proba_positive_wordBinary, proba_positiveBinary )
    negativeproba = calculateProbaBayes(words,proba_negative_wordBinary, proba_negativeBinary )
    
    positiveproba = positiveproba /(positiveproba+negativeproba)
    negativeproba = negativeproba /(positiveproba+negativeproba)
    
    if positiveproba >=  negativeproba :
        return "1"
     # neutralproba + positiveproba = notnegativeproba 
    return "0"
    

def processTest(data,outputFilePathEmotion,outputFilePathValenceBinary,outputFilePathValenceTriple):
    fileValenceBinary = open(outputFilePathValenceBinary, 'w+')
    fileValenceTriple = open(outputFilePathValenceTriple, 'w+')
    fileEmotion = open(outputFilePathEmotion, 'w+')
    for id, phrase in data.items():
        phrase = preprocessPhrase(phrase)
        fileValenceBinary.write('{} {}\n'.format(id, processPhraseValenceBinary(phrase)))
        fileValenceTriple.write('{} {}\n'.format(id, processPhraseValenceTriple(phrase)))
        fileEmotion.write('{} {}\n'.format(id, processPhraseEmotions(phrase)))

def training(training_data,FileEmotion,FileBinaryValence,FileTripleValence):
    valenceBinaryDict = convert_csvValence_to_dict(FileBinaryValence)
    valenceTripleDict = convert_csvValence_to_dict(FileTripleValence)
    emotionDict = convert_csvEmotions_to_dict(FileEmotion)
    
    trainingBinaryValence(training_data,valenceBinaryDict)   
    trainingTripleValence(training_data,valenceTripleDict) 
    trainingEmotion(training_data,emotionDict)

        
training_data = {}
test_data ={}

loadXmlData('datasets/AffectiveText.trial/affectivetext_trial.xml', training_data)

valenceFile = "datasets/AffectiveText.trial/affectivetext_trial.valence.gold"
emotionFile = "datasets/AffectiveText.trial/affectivetext_trial.emotions.gold"

convertEmotionFile(emotionFile,"datasets/AffectiveText.trial/emotion.csv")

convertValenceFiles(valenceFile,"datasets/AffectiveText.trial/valenceBinary.csv","datasets/AffectiveText.trial/valenceTriple.csv")

training(training_data,"datasets/AffectiveText.trial/emotion.csv","datasets/AffectiveText.trial/valenceBinary.csv","datasets/AffectiveText.trial/valenceTriple.csv")
 

loadXmlData('datasets/AffectiveText.test/affectivetext_test.xml', test_data)

outputfileTestEmotion = "results/test_emotions_Naive_Bayes.gold"

outputfileTestValenceBinary = "results/test_valenceBinary_Naive_Bayes.gold"

outputfileTestValenceTriple = "results/test_valenceTriple_Naive_Bayes.gold"

processTest(test_data,outputfileTestEmotion,outputfileTestValenceBinary,outputfileTestValenceTriple)        
