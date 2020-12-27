
results =[]

def convertFile(file1, file2):
    global results
    f = open(file1, 'r') 
    results = [int(word) for line in f for word in line.split()]
    f.close()
    file = open(file2, 'w')
    i = 0
    while i < len(results)-1:
        value = results[i+1]
        index = results[i]
        if value > 0:
            file.write(str(index) +" 1\n")
        if value < 0:
            file.write(str(index) +" -1\n")
        if value == 0:
            file.write(str(index) +" 0\n")
       
        i+=2
    file.close()


fTrial = "datasets/AffectiveText.trial/affectivetext_trial.valence.gold"
fileTrial = "datasets/AffectiveText.trial/valence.gold"

fTest = "datasets/AffectiveText.test/affectivetext_test.valence.gold"
fileTest = "datasets/AffectiveText.test/valence.gold"

resultTrial = "results/trial-valence.gold"
resultTest = "results/test-valence.gold"

resultTrial2 = "resultswithwords/trial-valence_positive.gold"
resultTest2 = "resultswithwords/test-valence_positive.gold"


convertFile(fTrial,fileTrial) 
convertFile(fTest,fileTest) 


def compaireFiles(file1,file2):
    f1 = open(file1, 'r') 
    result1 = [int(word) for line in f1 for word in line.split()]
    f1.close()
    
    f2 = open(file2, 'r') 
    result2 = [int(word) for line in f2 for word in line.split()]
    f2.close()
    erreur =0
    i = 0
    while i < len(result1):
        if result1[i] != result2[i]:
            if abs(result1[i]-result2[i]) == 1 :
                erreur += 0.7
            else:
                erreur +=1
        i += 1
    pourcentageErreur = erreur*100/len(result1)
    pourcentageCorrect = 100 - pourcentageErreur
    print("in file "+file2 +" erreur  is "+ str(pourcentageErreur) +" % and correct is "+ str(pourcentageCorrect))
    
compaireFiles(fileTrial,resultTrial2) 
compaireFiles(fileTest,resultTest2)   
 
compaireFiles(fileTrial,resultTrial)  
compaireFiles(fileTest,resultTest)  
    
"""
Erreur fatal
in file resultsWithpositive/trial-valence_positive.gold erreur  is 7.6 % and correct is 92.4
in file resultsWithpositive/test-valence_positive.gold erreur  is 8.55 % and correct is 91.45
in file results/trial-valence.gold erreur  is 9.8 % and correct is 90.2
in file results/test-valence.gold erreur  is 10.25 % and correct is 89.75

Erreur 

in file resultsWithpositive/trial-valence_positive.gold erreur  is 22.6 % and correct is 77.4
in file resultsWithpositive/test-valence_positive.gold erreur  is 25.75 % and correct is 74.25
in file results/trial-valence.gold erreur  is 24.6 % and correct is 75.4
in file results/test-valence.gold erreur  is 24.6 % and correct is 75.4


Eureur non fatal is count 0.7 instead of 1 
in file resultsWithpositive/trial-valence_positive.gold erreur  is 18.1% and correct is 81.89
in file resultsWithpositive/test-valence_positive.gold erreur  is 20.5 % and correct is 79.4
in file results/trial-valence.gold erreur  is 20.16 % and correct is 79.83
in file results/test-valence.gold erreur  is 20.29 % and correct is 79.7


"""



"""
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
    limit = total_count*0.05
    pourcentageNeg = negative_count*100/len(words)
    pourcentagePos = positive_count*100/len(words)
    pourcentageNeu = 100 -(pourcentageNeg + pourcentagePos)
    if pourcentageNeu > pourcentageNeg and pourcentageNeu > pourcentagePos or pourcentageNeg == pourcentagePos  :
        return "0"
    elif  pourcentageNeg > pourcentagePos :
        return "-1"
    else:
        return "1"

"""