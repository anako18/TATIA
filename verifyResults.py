
results =[]
lenth = 0

lines =[]
limit = 10
def binaryResult(value, limit):
    if value < limit:
        return "0"
    else:
        return "1"
        
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
        f.write( str(result[0]) +" ")
        while i < len(result):
           f.write( binaryResult( result[i],limit) +" ")
           i+=1
        f.write("\n")  
        
    f.close()

def convertValenceFile(file1, file2):
    global results
    lenth = 0
    f = open(file1, 'r') 
    results = [int(word) for line in f for word in line.split()]
    f.close()
    file = open(file2, 'w')
    i = 0
    while i < len(results):
        value = results[i+1]
        index = results[i]
        if value > 0:
            file.write(str(index) +" 1\n")
        if value < 0:
            file.write(str(index) +" -1\n")
        if value == 0:
            file.write(str(index) +" 0\n")
        lenth +=1
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

emotionTrial = "datasets/AffectiveText.trial/affectivetext_trial.emotions.gold"
fileemotionTrial = "datasets/AffectiveText.trial/emotions.gold"

emotionTest = "datasets/AffectiveText.test/affectivetext_test.emotions.gold"
fileemotionTest = "datasets/AffectiveText.test/emotions.gold"


emotionTrialResult = "results/trial-emotions.gold"
emotionTestResult = "results/test-emotions.gold"

convertEmotionFile(emotionTrial,fileemotionTrial)   
convertEmotionFile(emotionTestResult,fileemotionTest) 


convertValenceFile(fTrial,fileTrial) 
convertValenceFile(fTest,fileTest) 

def compaireValenceFiles(file1,file2):
    f1 = open(file1, 'r') 
    result1 = [word for line in f1 for word in line.split()]
    f1.close()
    
    f2 = open(file2, 'r') 
    result2 = [word for line in f2 for word in line.split()]
    f2.close()
    erreur =0
    i = 0
    while i < len(result1):
        if result1[i+1].strip() != result2[i+1].strip():
            erreur +=1
        i += 2
    pourcentageErreur = erreur*100/(len(result1) /2)
    pourcentageCorrect = 100 - pourcentageErreur
    print("in file "+file2 +" erreur  is "+ str(pourcentageErreur) +" % and correct is "+ str(pourcentageCorrect) +"% nb of erreurs "+str(erreur) + " / " + str(int(len(result1)/2)))
    

compaireValenceFiles(fileTrial,resultTrial)  
compaireValenceFiles(fileTest,resultTest)  


def compaireEmotionFiles(file1,file2):
    f1 = open(file1, 'r') 
    result1 = [line for line in f1 ]
    f1.close()
    
    f2 = open(file2, 'r') 
    result2 = [line for line in f2]
    f2.close()        
    erreur =0
    i = 0
    total = 0
    while i < len(result1):
        line1 = [l for l in result1[i].split() ]
        line2 = [l for l in result2[i].split() ]
        j =1
        while j < len(line1):
            if line1[j].strip() != line2[j].strip():
                erreur +=1
            j += 1
            total += 1
        i+=1
    pourcentageErreur = erreur*100/total
    pourcentageCorrect = 100 - pourcentageErreur
    print("in file "+file2 +" erreur  is "+ str(pourcentageErreur) +" % and correct is "+ str(pourcentageCorrect) +"% nb of erreurs "+str(erreur) + " / " + str(total))
        


compaireEmotionFiles(fileemotionTrial,emotionTrialResult)    

compaireEmotionFiles(fileemotionTest,emotionTestResult)

