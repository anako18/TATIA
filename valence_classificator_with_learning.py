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

training_data = []
training_labels = []
training_ids = []

test_data = []
test_labels = []
test_ids = []

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
def getLabeBinary(value):
    global pos
    global neg
    global neut
    if (value > 0):
        return "positive"
    else:
        return "negative"

#fill in ids, data and labels
def loadXmlData(xmlFilePath, labelsFilePath, ids, data, labels):
    xmldoc = minidom.parse(xmlFilePath)
    nodelist = xmldoc.getElementsByTagName('instance')

    f = open(labelsFilePath, 'r')
    result_labels = f.readlines()
    i = 0
    for node in nodelist:
        ids.append(node.getAttribute('id'))
        data.append(preprocessPhrase(node.firstChild.nodeValue.lower()))
        #3 classes: positive, negative and neutral
        labels.append(getLabelTriple(int(result_labels[i].split()[1]), 20))
        
        #2 classes: positive and negative
        #labels.append(getLabeBinary(int(result_labels[i].split()[1])))
        i+=1

loadXmlData(
    'datasets/AffectiveText.trial/affectivetext_trial.xml',
    'datasets/AffectiveText.trial/affectivetext_trial.valence.gold', 
    training_ids, 
    training_data, 
    training_labels
)

loadXmlData(
    'datasets/AffectiveText.test/affectivetext_test.xml',
    'datasets/AffectiveText.test/affectivetext_test.valence.gold', 
    test_ids, 
    test_data, 
    test_labels
)

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