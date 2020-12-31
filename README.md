# TATIA project: Sentiment analysis and emotions recognition
Test and training datas: https://web.eecs.umich.edu/~mihalcea/affectivetext/

# Classificators for the following problems:

### 1 - Binary sentiment analysis (determines if the text sample is positive or negative)
Datatset: https://www.kaggle.com/harshaiitj08/positive-and-negative-words

### 2 - Ternary sentiment analysis (determines if the text sample is positive, negative or neutral)
Datatset: https://www.kaggle.com/harshaiitj08/positive-and-negative-words

### 3- Emotions recognition (determines if each of the six basic emotions (anger disgust fear joy sadness surprise)
Datatset: https://web.eecs.umich.edu/~mihalcea/affectivetext/

# Implemented systems:

* Rule-based system
* Naive Bayes classificator
* SVC with rbf kernel
* SVC with polynomial kernel
* SVC with sigmoid kernel
* SVC with linear kernel
* LinearSVC

Contributors: Meryem BOUFALAH and Anastasiia KOZLOVA

# Description of files:

The results are stored in /result folder
The datasets are stored in /datasets folder

1) rule_based_classificator.py - Rule-based classificator
To run:
```sh
$ python3 rule_based_classificator.py 
```
2) Naive_Bayes_Classifier.py - Naive Bayes classificator
To run:
```sh
$ python3 Naive_Bayes_Classifier.py 
```
3) classificator_with_learning.py - SVC classificators
In code we can set classifier_number in able to choose which SVC to run or uncomment the part where we run all the classificators. All the classification reports are printed and the results of last run classificator is written to file.
To run:
```sh
$ python3 classificator_with_learning.py
```
4) buildClassificationReports.py - build classification reports for all the result files
To run:
```sh
$ python3 buildClassificationReports.py
```