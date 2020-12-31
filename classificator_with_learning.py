import string

from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import xml.dom.minidom
from xml.dom import minidom

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

emotions_threshold = 20
valence_threshold = 30

training_data = []
training_ids = []

test_data = []
test_ids = []

valence_training_labels = []
valence_test_labels = []

valence_triple_training_labels = []
valence_triple_test_labels = []

training_anger_labels = []
training_disgust_labels = []
training_fear_labels = []
training_joy_labels = []
training_sadness_labels = []
training_surprise_labels = []

test_anger_labels = []
test_disgust_labels = []
test_fear_labels = []
test_joy_labels = []
test_sadness_labels = []
test_surprise_labels = []

result_valence_labels = []
result_triple_valence_labels = []

result_anger_labels = []
result_disgust_labels = []
result_fear_labels = []
result_joy_labels = []
result_sadness_labels = []
result_surprise_labels = []

def preprocessPhrase(phrase):
    #remove new lines
    phrase = phrase.replace('\n', '')

    #remove punctuation signs
    phrase = phrase.translate(str.maketrans('', '', string.punctuation))

    #tokenize
    tokens = word_tokenize(phrase)

     #remove stopwords
    filtered_words = [w for w in tokens if not w in stop_words]

    #lemmatization
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    res = ""
    for word in lemma_words:
        res+=word + " "
    return res

#if value > threshold then emotion exists
#threshold is in range [0, 100]
def emotionToBinary(value, threshold):
    if (value >= threshold):
        return 1
    else:
        return 0

#the valence is in [-100, 100]
#[-100, -threshold] -> negative
#[-threshold, threshold] -> neutral
#[threshold, 100] -> positive
def getLabelTriple(value, threshold):
    if (value >= threshold):
        return 1
    elif (value < threshold) and (value > -1*threshold):
        return 0
    else:
        return -1

#Positive or negative
def getLabelBinary(value):
    if (value > 0):
        return 1
    else:
        return 0

def loadXmlData(xmlFilePath, ids, data):
    xmldoc = minidom.parse(xmlFilePath)
    nodelist = xmldoc.getElementsByTagName('instance')

    for node in nodelist:
        ids.append(node.getAttribute('id'))
        data.append(preprocessPhrase(node.firstChild.nodeValue.lower()))

#Load labels for positive and negative classification
def loadValenceLabelsBinary(labelsFilePath, labels):
    f = open(labelsFilePath, 'r')
    file_labels = f.readlines()
    for label in file_labels:
        labels.append(getLabelBinary(int(label.split()[1])))
    f.close()

#Load labels for positive, neutral and negative classification
def loadValenceLabelsTriple(labelsFilePath, labels):
    f = open(labelsFilePath, 'r')
    file_labels = f.readlines()
    for label in file_labels:
        labels.append(getLabelTriple(int(label.split()[1]), valence_threshold))
    f.close()


#Load test labels for all emotions
def loadEmotionsLabelsTraining(labelsFilePath):
    f = open(labelsFilePath, 'r')
    file_labels = f.readlines()
    for label in file_labels:
        emotions = label.split()
        training_anger_labels.append(emotionToBinary(int(emotions[1]), emotions_threshold))
        training_disgust_labels.append(emotionToBinary(int(emotions[2]), emotions_threshold))
        training_fear_labels.append(emotionToBinary(int(emotions[3]), emotions_threshold))
        training_joy_labels.append(emotionToBinary(int(emotions[4]), emotions_threshold))
        training_sadness_labels.append(emotionToBinary(int(emotions[5]), emotions_threshold))
        training_surprise_labels.append(emotionToBinary(int(emotions[6]), emotions_threshold))
    f.close()

#Load test labels for all emotions
def loadEmotionsLabelsTest(labelsFilePath):
    f = open(labelsFilePath, 'r')
    file_labels = f.readlines()
    for label in file_labels:
        emotions = label.split()
        test_anger_labels.append(emotionToBinary(int(emotions[1]), emotions_threshold))
        test_disgust_labels.append(emotionToBinary(int(emotions[2]), emotions_threshold))
        test_fear_labels.append(emotionToBinary(int(emotions[3]), emotions_threshold))
        test_joy_labels.append(emotionToBinary(int(emotions[4]), emotions_threshold))
        test_sadness_labels.append(emotionToBinary(int(emotions[5]), emotions_threshold))
        test_surprise_labels.append(emotionToBinary(int(emotions[6]), emotions_threshold))
    f.close()

def writeValenceResultFile(test_ids, valence_result_file_path, labels):
    fileValence = open(valence_result_file_path, 'w+')
    for i in range(len(test_ids)):
        fileValence.write('{} {}\n'.format(test_ids[i], labels[i]))
    fileValence.close()

def writeEmotionsResultFile(test_ids, emotions_result_file_path):
    fileEmotions = open(emotions_result_file_path, 'w+')
    for i in range(len(test_ids)):
        fileEmotions.write('{} {} {} {} {} {} {}\n'.format(
            test_ids[i], 
            result_anger_labels[i], 
            result_disgust_labels[i],
            result_fear_labels[i],
            result_joy_labels[i],
            result_sadness_labels[i],
            result_surprise_labels[i]
        ))
    fileEmotions.close()

#classifier_to_save_to_run is a number of classifier to run
# 0 => SVM, kernel=rbf
# 1 => SVM kernel=poly
# 2 => SVM kernel=sigmoid
# 3 => SVM kernel=linear
# 4 => LinearSVC
def classify(training_data, test_data, training_labels, test_labels, classifier_to_run = 0):
    result_labels = []
    #vectorization with tfid
    vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=3, max_df = 0.8, sublinear_tf = True)
    train_vectors = vectorizer.fit_transform(training_data)
    test_vectors = vectorizer.transform(test_data)

    if classifier_to_run == 0:
        #SVM, kernel=rbf
        classifier_rbf = svm.SVC()
        classifier_rbf.fit(train_vectors, training_labels)
        prediction_rbf = classifier_rbf.predict(test_vectors)
        result_labels = prediction_rbf
        print("===========================SVC(kernel=rbf)===========================")
        print(confusion_matrix(test_labels, prediction_rbf))
        print(classification_report(test_labels, prediction_rbf))

    elif classifier_to_run == 1:
        #SVM kernel=poly
        classifier_poly = svm.SVC(kernel='poly')
        classifier_poly.fit(train_vectors, training_labels)
        prediction_poly = classifier_poly.predict(test_vectors)
        result_labels = prediction_poly
        print("=========================== SVC(kernel=poly) ===========================")
        print(confusion_matrix(test_labels, prediction_poly))
        print(classification_report(test_labels, prediction_poly))
    elif classifier_to_run == 2:
        #SVM kernel=sigmoid
        classifier_sigmoid = svm.SVC(kernel='sigmoid')
        classifier_sigmoid.fit(train_vectors, training_labels)
        prediction_sigmoid = classifier_sigmoid.predict(test_vectors)
        result_labels = prediction_sigmoid
        print("=========================== SVC(kernel=sigmoid) ===========================")
        print(confusion_matrix(test_labels, prediction_sigmoid))
        print(classification_report(test_labels, prediction_sigmoid))
    elif classifier_to_run == 3:
        #SVM kernel=linear
        classifier_linear = svm.SVC(kernel='linear')
        classifier_linear.fit(train_vectors, training_labels)
        prediction_linear = classifier_linear.predict(test_vectors)
        result_labels = prediction_linear
        print("=========================== SVC(kernel=linear) ===========================")
        print(confusion_matrix(test_labels, prediction_linear))
        print(classification_report(test_labels, prediction_linear))
    else:
        #LinearSVC
        classifier_liblinear = svm.LinearSVC()
        classifier_liblinear.fit(train_vectors, training_labels)
        prediction_liblinear = classifier_liblinear.predict(test_vectors)
        result_labels = prediction_liblinear
        print("=========================== LinearSVC ===========================")
        print(confusion_matrix(test_labels, prediction_liblinear))
        print(classification_report(test_labels, prediction_liblinear))
    return result_labels
    
valence_test_file = "datasets/AffectiveText.test/affectivetext_test.valence.gold"
emotions_test_file = "datasets/AffectiveText.test/affectivetext_test.emotions.gold"

valence_trial_file = "datasets/AffectiveText.trial/affectivetext_trial.valence.gold"
emotions_trial_file = "datasets/AffectiveText.trial/affectivetext_trial.emotions.gold"

valence_SVM_result_file = 'results/test_valence_SVM.gold'
valence_triple_SVM_result_file = 'results/test_triple_valence_SVM.gold'
emotions_SVM_result_file = 'results/test_emotions_SVM.gold'

#Load common training and testing data
loadXmlData('datasets/AffectiveText.trial/affectivetext_trial.xml', training_ids, training_data)
loadXmlData('datasets/AffectiveText.test/affectivetext_test.xml', test_ids, test_data)

#Valence binary (positive and negative)
loadValenceLabelsBinary(valence_trial_file, valence_training_labels)
loadValenceLabelsBinary(valence_test_file, valence_test_labels)
print('================================================================= Sentiment analysis (binary) =================================================================')
result_valence_labels = classify(training_data, test_data, valence_training_labels, valence_test_labels)

#Valence triple (positive, negative and neutral)
loadValenceLabelsTriple(valence_trial_file, valence_triple_training_labels)
loadValenceLabelsTriple(valence_test_file, valence_triple_test_labels)
print('================================================================= Sentiment analysis (triple) =================================================================')
result_triple_valence_labels = classify(training_data, test_data, valence_triple_training_labels, valence_triple_test_labels)

#Emotions recognition
print('==================================================================== Emotions recognition =====================================================================')
loadEmotionsLabelsTraining(emotions_trial_file)
loadEmotionsLabelsTest(emotions_test_file)
print("=============================================================")
print("=========================== Anger ===========================")
print("=============================================================")
result_anger_labels = classify(training_data, test_data, training_anger_labels, test_anger_labels)

print("=============================================================")
print("=========================== Disgust =========================")
print("=============================================================")
result_disgust_labels = classify(training_data, test_data, training_disgust_labels, test_disgust_labels)

print("=============================================================")
print("=========================== Fear =========================")
print("=============================================================")
result_fear_labels = classify(training_data, test_data, training_fear_labels, test_fear_labels)

print("=============================================================")
print("=========================== Joy =========================")
print("=============================================================")
result_joy_labels = classify(training_data, test_data, training_joy_labels, test_joy_labels)

print("=============================================================")
print("=========================== Sadness =========================")
print("=============================================================")
result_sadness_labels = classify(training_data, test_data, training_sadness_labels, test_sadness_labels)

print("=============================================================")
print("=========================== Surprise =========================")
print("=============================================================")
result_surprise_labels = classify(training_data, test_data, training_surprise_labels, test_surprise_labels)

writeEmotionsResultFile(test_ids, emotions_SVM_result_file)
writeValenceResultFile(test_ids, valence_SVM_result_file, result_valence_labels)
writeValenceResultFile(test_ids, valence_triple_SVM_result_file, result_triple_valence_labels)