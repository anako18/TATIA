import string

from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import classification_report

import xml.dom.minidom
from xml.dom import minidom

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer() 

emotions_threshold = 30
sentiments_threshold = 20

training_data = []
training_ids = []

test_data = []
test_ids = []

valence_training_labels = []
valence_test_labels = []

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

pos = 0
neg = 0
neut = 0

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
    global pos
    global neg
    global neut
    if (value >= threshold):
        pos+=1
        return "positive"
    elif (value < threshold) and (value > -1*threshold):
        neut+=1
        return "neutral"
    else:
        neg+=1
        return "negative"

#Positive or negative
def getLabelBinary(value):
    global pos
    global neg
    global neut
    if (value > 0):
        return "positive"
    else:
        return "negative"

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

#Load labels for positive, neutral and negative classification
def loadValenceLabelsTriple(labelsFilePath, labels):
    f = open(labelsFilePath, 'r')
    file_labels = f.readlines()
    for label in file_labels:
        labels.append(getLabelTriple(int(label.split()[1]), sentiments_threshold))


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

def classify(training_data, test_data, training_labels, test_labels):
    #vectorization with tfid
    vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=2, max_df = 0.8, sublinear_tf = True)
    train_vectors = vectorizer.fit_transform(training_data)
    test_vectors = vectorizer.transform(test_data)

    #SVM, kernel=rbf
    classifier_rbf = svm.SVC()
    classifier_rbf.fit(train_vectors, training_labels)

    #SVM, kernel=rbf
    classifier_rbf = svm.SVC()
    classifier_rbf.fit(train_vectors, training_labels)
    prediction_rbf = classifier_rbf.predict(test_vectors)

    #SVM kernel=poly
    classifier_poly = svm.SVC(kernel='poly')
    classifier_poly.fit(train_vectors, training_labels)
    prediction_poly = classifier_poly.predict(test_vectors)

    #SVM kernel=sigmoid
    classifier_sigmoid = svm.SVC(kernel='sigmoid')
    classifier_sigmoid.fit(train_vectors, training_labels)
    prediction_sigmoid = classifier_sigmoid.predict(test_vectors)

    #SVM kernel=linear
    classifier_linear = svm.SVC(kernel='linear')
    classifier_linear.fit(train_vectors, training_labels)
    prediction_linear = classifier_linear.predict(test_vectors)

    #LinearSVC
    classifier_liblinear = svm.LinearSVC()
    classifier_liblinear.fit(train_vectors, training_labels)
    prediction_liblinear = classifier_liblinear.predict(test_vectors)

    print("===========================SVC(kernel=rbf)===========================")
    print(classification_report(test_labels, prediction_rbf))

    print("=========================== SVC(kernel=poly) ===========================")
    print(classification_report(test_labels, prediction_poly))

    print("=========================== SVC(kernel=sigmoid) ===========================")
    print(classification_report(test_labels, prediction_sigmoid))

    print("=========================== SVC(kernel=linear) ===========================")
    print(classification_report(test_labels, prediction_linear))

    print("=========================== LinearSVC ===========================")
    print(classification_report(test_labels, prediction_liblinear))
        

#0 => binary valence
#1 => triple valence
#2 => emotions
mode = 2

loadXmlData('datasets/AffectiveText.trial/affectivetext_trial.xml', training_ids, training_data)
loadXmlData('datasets/AffectiveText.test/affectivetext_test.xml', test_ids, test_data)

if (mode == 0):
    loadValenceLabelsBinary('datasets/AffectiveText.trial/affectivetext_trial.valence.gold', valence_training_labels)
    loadValenceLabelsBinary('datasets/AffectiveText.test/affectivetext_test.valence.gold', valence_test_labels)
    print('================================================================= Sentiment analysis (binary) =================================================================')
    classify(training_data, test_data, valence_training_labels, valence_test_labels)
elif (mode == 1):
    loadValenceLabelsTriple('datasets/AffectiveText.trial/affectivetext_trial.valence.gold', valence_training_labels)
    loadValenceLabelsTriple('datasets/AffectiveText.test/affectivetext_test.valence.gold', valence_test_labels)
    print('================================================================= Sentiment analysis (triple) =================================================================')
    classify(training_data, test_data, valence_training_labels, valence_test_labels)
else:
    print('==================================================================== Emotions recognition =====================================================================')
    loadEmotionsLabelsTraining('datasets/AffectiveText.trial/affectivetext_trial.emotions.gold')
    loadEmotionsLabelsTest('datasets/AffectiveText.test/affectivetext_test.emotions.gold')
    print("=============================================================")
    print("=========================== Anger ===========================")
    print("=============================================================")
    classify(training_data, test_data, training_anger_labels, test_anger_labels)

    print("=============================================================")
    print("=========================== Disgust =========================")
    print("=============================================================")
    classify(training_data, test_data, training_disgust_labels, test_disgust_labels)

    print("=============================================================")
    print("=========================== Fear =========================")
    print("=============================================================")
    classify(training_data, test_data, training_fear_labels, test_fear_labels)

    print("=============================================================")
    print("=========================== Joy =========================")
    print("=============================================================")
    classify(training_data, test_data, training_joy_labels, test_joy_labels)

    print("=============================================================")
    print("=========================== Sadness =========================")
    print("=============================================================")
    classify(training_data, test_data, training_sadness_labels, test_sadness_labels)

    print("=============================================================")
    print("=========================== Surprise =========================")
    print("=============================================================")
    classify(training_data, test_data, training_surprise_labels, test_surprise_labels)