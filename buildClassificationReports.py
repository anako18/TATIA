from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

threshold_emotions = 20
threshold_valence = 30

def toBinary(value, threshold):
    if (value >= threshold):
        return 1
    else:
        return 0

def toTriple(value, threshold):
    if (value >= threshold):
        return 1
    elif (value < threshold) and (value > -1*threshold):
        return 0
    else:
        return -1

def compareValenceClassificationReport(testFilePath, resultFilePath, triple = 0):
    f1 = open(testFilePath, 'r') 
    firstFileValence = [line for line in f1 ]
    f1.close()
    f2 = open(resultFilePath, 'r') 
    secondFileValence = [line for line in f2]
    f2.close()

    errorsCount = 0
    total = len(firstFileValence)

    test_labels = []
    result_labels = []

    for i in range(len(firstFileValence)):
        coefs1 = firstFileValence[i].split()
        coefs2 = secondFileValence[i].split()
        testValenceValue = toBinary(int(coefs1[1]), 0)
        if triple == 1:
            testValenceValue = toTriple(int(coefs1[1]), threshold_valence)    
        resultValenceValue = int(coefs2[1])
        if testValenceValue != resultValenceValue:
            errorsCount+=1
        test_labels.append(testValenceValue)
        result_labels.append(resultValenceValue)
    
    print(confusion_matrix(test_labels,result_labels))
    print(classification_report(test_labels, result_labels))
    
    correctAnswersPercent = 100 - errorsCount*100/total
    print("Total samples: {}, total errors count: {}, correct answers percentage: {}%".format(total, errorsCount, correctAnswersPercent))

def compaireEmotionClassificationReport(testFilePath, resultFilePath):
    f1 = open(testFilePath, 'r') 
    firstFileEmotions = [line for line in f1 ]
    f1.close()
    f2 = open(resultFilePath, 'r') 
    secondtFileEmotions = [line for line in f2]
    f2.close()

    errorsCount = 0
    total = len(firstFileEmotions)*6

    test_labels = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: []
    }
    result_labels = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: []
    }

    emotions = {
        1: "anger",
        2: "disgust",
        3: "fear",
        4: "joy",
        5: "sadness",
        6: "surprise" 
    }
    for i in range(len(firstFileEmotions)):
        
        coefs1 = firstFileEmotions[i].split()
        coefs2 = secondtFileEmotions[i].split()
        for key in emotions:
            testEmotionValue = toBinary(int(coefs1[key]), threshold_emotions)
            resultEmotionValue = int(coefs2[2])
            if testEmotionValue != resultEmotionValue:
                errorsCount+=1
            test_labels[key].append(testEmotionValue)
            result_labels[key].append(resultEmotionValue)
    
    for key, value in emotions.items():
        print("================================== " + value + " ==================================")
        print(confusion_matrix(test_labels[key], result_labels[key]))
        print(classification_report(test_labels[key], result_labels[key]))
    
    correctAnswersPercent = 100 - errorsCount*100/total
    print("Total samples: {}, total errors count: {}, correct answers percentage: {}%".format(total, errorsCount, correctAnswersPercent))
    
valenceFile = "datasets/AffectiveText.test/affectivetext_test.valence.gold"
emotionFile = "datasets/AffectiveText.test/affectivetext_test.emotions.gold"

valence_SVM_result_file = 'results/test_valence_SVM.gold'
valence_triple_SVM_result_file = 'results/test_triple_valence_SVM.gold'

valenceTriple_NaiveBayes_result_file = "results/test_valenceTriple_Naive_Bayes.gold"
valenceBinary_NaiveBayes_result_file = "results/test_valenceBinary_Naive_Bayes.gold"
valence_RuleBased_result_file = "results/test_valence_rule_based.gold"
valence_triple_RuleBased_result_file = "results/test_triple_valence_rule_based.gold"

emotions_SVM_result_file = 'results/test_emotions_SVM.gold'
emotions_NaiveBayes_result_file = "results/test_emotions_Naive_Bayes.gold"
emotions_RuleBased_result_file = "results/test_emotions_rule_based.gold"
print("===============================================================================================================")
print("============================================== Rules Based ====================================================")
print("===============================================================================================================")
print("====================================== Valence binary =============================================")
compareValenceClassificationReport(valenceFile, valence_RuleBased_result_file)
print("====================================== Valence triple =============================================")
compareValenceClassificationReport(valenceFile, valence_triple_RuleBased_result_file, 1)
print("======================================== Emotions =================================================")
compaireEmotionClassificationReport(emotionFile, emotions_RuleBased_result_file)

print("\n===============================================================================================================")
print("================================================ SVM ==========================================================")
print("===============================================================================================================")
print("====================================== Valence binary =============================================")
compareValenceClassificationReport(valenceFile, valence_SVM_result_file)
print("====================================== Valence triple =============================================")
compareValenceClassificationReport(valenceFile, valence_triple_SVM_result_file, 1)
print("======================================== Emotions =================================================")
compaireEmotionClassificationReport(emotionFile, emotions_SVM_result_file)


print("\n===============================================================================================================")
print("============================================= Naive Bayes =======================================================")
print("=================================================================================================================")
print("====================================== Valence binary =============================================")
compareValenceClassificationReport(valenceFile, valenceBinary_NaiveBayes_result_file)
print("================================= Valence triple ========================================")
compareValenceClassificationReport(valenceFile, valenceTriple_NaiveBayes_result_file,1)
print("===================================== Emotions ==========================================")
compaireEmotionClassificationReport(emotionFile, emotions_NaiveBayes_result_file)

