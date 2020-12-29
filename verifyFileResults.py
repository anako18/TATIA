from sklearn.metrics import classification_report

threshold_emotions = 20

def emotionToBinary(value):
    if (value >= threshold_emotions):
        return 1
    else:
        return 0

def compaireEmotionClassificationReport(testFilePath, resultFilePath):
    f1 = open(testFilePath, 'r') 
    firstFileEmotions = [line for line in f1 ]
    f1.close()
    f2 = open(resultFilePath, 'r') 
    secondtFileEmotions = [line for line in f2]
    f2.close()

    test_anger_labels = []
    test_disgust_labels = []
    test_fear_labels = []
    test_joy_labels = []
    test_sadness_labels = []
    test_surprise_labels = []
    
    result_anger_labels = []
    result_disgust_labels = []
    result_fear_labels = []
    result_joy_labels = []
    result_sadness_labels = []
    result_surprise_labels = []
    for i in range(len(firstFileEmotions)):
        coefs1 = firstFileEmotions[i].split()
        coefs2 = secondtFileEmotions[i].split()

        test_anger_labels.append(emotionToBinary(int(coefs1[1])))
        result_anger_labels.append(int(coefs2[1]))

        test_disgust_labels.append(emotionToBinary(int(coefs1[2])))
        result_disgust_labels.append(int(coefs2[2]))

        test_fear_labels.append(emotionToBinary(int(coefs1[3])))
        result_fear_labels.append(int(coefs2[3]))

        test_joy_labels.append(emotionToBinary(int(coefs1[4])))
        result_joy_labels.append(int(coefs2[4]))

        test_sadness_labels.append(emotionToBinary(int(coefs1[5])))
        result_sadness_labels.append(int(coefs2[5]))

        test_surprise_labels.append(emotionToBinary(int(coefs1[6])))
        result_surprise_labels.append(int(coefs2[6]))
    print("==============ANGER===========")
    print(classification_report(test_anger_labels, result_anger_labels))
    
    print("==============DISGUST===========")
    print(classification_report(test_disgust_labels, result_disgust_labels))

    print("==============FEAR===========")
    print(classification_report(test_fear_labels, result_fear_labels))

    print("==============JOY===========")
    print(classification_report(test_joy_labels, result_joy_labels))

    print("==============SADNESS===========")
    print(classification_report(test_sadness_labels, result_sadness_labels))

    print("==============SURPRISE===========")
    print(classification_report(test_surprise_labels, result_surprise_labels))

def compaireEmotionFiles(testFilePath, resultFilePath):
    f1 = open(testFilePath, 'r') 
    firstFileEmotions = [line for line in f1 ]
    f1.close()
    f2 = open(resultFilePath, 'r') 
    secondtFileEmotions = [line for line in f2]
    f2.close()

    totals = [0,0,0,0,0,0]
    errors = [0,0,0,0,0,0]
    totalAccuracy = 0
    totalEmotions = len(firstFileEmotions)
    totalErrors = 0
    for i in range(len(firstFileEmotions)):
        coefs1 = firstFileEmotions[i].split()
        coefs2 = secondtFileEmotions[i].split()
        #(coefs1)
        #print(coefs2)
        for j in range(1,7):
            coefs1[j] = emotionToBinary(int(coefs1[j]))
            if coefs1[j] == 1:
                totals[j-1]+=1
            if coefs1[j] != int(coefs2[j]):
                errors[j-1]+=1
    for i in range(6):
        #totalEmotions+=totals[i]
        totalErrors+=errors[i]
        pourcentageErreur = errors[i]*100/totalEmotions
        pourcentageCorrect = 100 - pourcentageErreur
        print("Emotion {} accuracy is {}, errors are {}, total is {}".format(i, pourcentageCorrect, errors[i], totalEmotions))
    totalAccuracy = 100 - totalErrors*100/(totalEmotions*6)
    print("Total emotion results: accuracy is {}, errors {}/{}".format(totalAccuracy, totalErrors, totalEmotions))

compaireEmotionClassificationReport('datasets/AffectiveText.test/affectivetext_test.emotions.gold','results/test-emotions.gold')  