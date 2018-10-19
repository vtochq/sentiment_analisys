#import tensorflow as tf
from tensorflow import keras
import pandas as pd
#import codecs
import my_lib
import datetime
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from textblob import TextBlob
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from time import time

from bs4 import BeautifulSoup
import re

from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'html.parser')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Zа-яА-Я_]", " ", clean)
    lower_case = letters_only.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()

infprefix = "100_"
infprefix = ""
SEED = 20

print("Loading Positive ", datetime.datetime.now())
dataset = pd.DataFrame()

data_raw = pd.read_csv(infprefix+'positive.csv', header=None, encoding='utf8', sep=';')[3]
dataset['Words'] = data_raw
dataset['Cat'] = 1


print("Loading Negative ", datetime.datetime.now())
temp = pd.DataFrame()

data_raw = pd.read_csv(infprefix+'negative.csv', header=None, encoding='utf8', sep=';')[3]
temp['Words'] = data_raw
temp['Cat'] = 0

raw_dataset = dataset.append(temp, ignore_index=True)

#train_data = raw_dataset['Words'].apply(my_lib.splitstring)
train_data = raw_dataset['Words'].str.lower().values

train_data_cleaned = []
for i in range(0,len(train_data)):
    train_data_cleaned.append(tweet_cleaner(train_data[i]))

train_data = train_data_cleaned;

#rawtrain_data = dataset['Words']
train_labels = raw_dataset['Cat'].values

# print(dataset)
'''
print("Creating dictionary ", datetime.datetime.now())
words = my_lib.WordsDic(train_data)

print("Replace words to digits ", datetime.datetime.now())
train_data = my_lib.WordReplace(train_data, words)

print ("Padding matrix ", datetime.datetime.now())
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=0,\
    padding='post', maxlen=256)  # , maxlen=256

val_size = round(len(train_data)*0.1)  # 10% of train_data
'''

x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(train_data, train_labels, test_size=.1, random_state=SEED)

x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)


print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_train), (len(x_train[y_train == 0]) / (len(x_train)*1.))*100, (len(x_train[y_train == 1]) / (len(x_train)*1.))*100))
print("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_validation), (len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100, (len(x_validation[y_validation == 1]) / (len(x_validation)*1.))*100))
print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_test), (len(x_test[y_test == 0]) / (len(x_test)*1.))*100, (len(x_test[y_test == 1]) / (len(x_test)*1.))*100))

'''
# x_val = train_data[:3].reset_index(drop=True)
x_val = train_data[:val_size]
partial_x_train = train_data[val_size:]

y_val = train_labels[:val_size]
partial_y_train = train_labels[val_size:]
'''


tbresult = [TextBlob(i).sentiment.polarity for i in x_validation]
tbpred = [0 if n < 0 else 1 for n in tbresult]
conmat = np.array(confusion_matrix(y_validation, tbpred, labels=[1,0]))

confusion = pd.DataFrame(conmat, index=['positive', 'negative'],
                         columns=['predicted_positive','predicted_negative'])


print("Accuracy Score: {0:.2f}%".format(accuracy_score(y_validation, tbpred)*100))
print( "-"*80)
print( "Confusion Matrix\n")
print( confusion)
print( "-"*80)
print( "Classification Report\n")
print( classification_report(y_validation, tbpred))


def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
    if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
    t0 = time()
    sentiment_fit = pipeline.fit(x_train, y_train)
    y_pred = sentiment_fit.predict(x_test)
    train_test_time = time() - t0
    accuracy = accuracy_score(y_test, y_pred)
    print("null accuracy: {0:.2f}%".format(null_accuracy*100))
    print("accuracy score: {0:.2f}%".format(accuracy*100))
    if accuracy > null_accuracy:
        print("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100))
    elif accuracy == null_accuracy:
        print("model has the same accuracy with the null accuracy")
    else:
        print("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100))
    print("train and test time: {0:.2f}s".format(train_test_time))
    print("-"*80)
    return accuracy, train_test_time

cvec = CountVectorizer()
lr = LogisticRegression(solver='lbfgs', max_iter=1000) #solver='lbfgs'
n_features = np.arange(10000,100001,1000)

import nltk.corpus
stopwords = nltk.corpus.stopwords.words('russian')

from collections import Counter
my_stopwords=Counter(" ".join(train_data).split()).most_common(100)
my_stopwords=[item[0] for item in my_stopwords]

both_stopwords = list(set(my_stopwords+stopwords))

stop_words = ''

def nfeature_accuracy_checker(vectorizer=cvec, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=lr):
    result = []
    print((classifier))
    print("\n")
    for n in n_features:
        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        print("Validation result for {} features".format(n))
        nfeature_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
        result.append((n,nfeature_accuracy,tt_time))
    return result


print("RESULT FOR UNIGRAM WITHOUT STOP WORDS\n")
feature_result_wosw = nfeature_accuracy_checker()

print("RESULT FOR UNIGRAM WITH NLTK STOP WORDS\n")
feature_result_wnltksw = nfeature_accuracy_checker(stop_words=stopwords)

print("RESULT FOR UNIGRAM WITH MY STOP WORDS\n")
feature_result_wmysw = nfeature_accuracy_checker(stop_words=my_stopwords)

print("RESULT FOR UNIGRAM WITH BOTH STOP WORDS\n")
feature_result_wbothsw = nfeature_accuracy_checker(stop_words=both_stopwords)


import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

nfeatures_plot_ug_wosw = pd.DataFrame(feature_result_wosw,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ug_wnltksw = pd.DataFrame(feature_result_wnltksw,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ug_wmysw = pd.DataFrame(feature_result_wmysw,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ug_wbothsw = pd.DataFrame(feature_result_wbothsw,columns=['nfeatures','validation_accuracy','train_test_time'])
plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_ug_wosw.nfeatures, nfeatures_plot_ug_wosw.validation_accuracy, label='without stop words')
plt.plot(nfeatures_plot_ug_wnltksw.nfeatures, nfeatures_plot_ug_wnltksw.validation_accuracy,label='with nltk stop words')
plt.plot(nfeatures_plot_ug_wmysw.nfeatures, nfeatures_plot_ug_wmysw.validation_accuracy,label='with my stop words')
plt.plot(nfeatures_plot_ug_wbothsw.nfeatures, nfeatures_plot_ug_wbothsw.validation_accuracy,label='with both stop words')
plt.title("Without stop words VS With stop words (Unigram): Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()
plt.show()


print("RESULT FOR BIGRAM WITHOUT STOP WORDS\n")
feature_result_bg = nfeature_accuracy_checker(ngram_range=(1, 2), stop_words=[''], n_features=n_features)
print("RESULT FOR TRIGRAM WITHOUT STOP WORDS\n")
feature_result_tg = nfeature_accuracy_checker(ngram_range=(1, 3), stop_words=[''], n_features=n_features)
print("RESULT FOR FOURGRAM WITHOUT STOP WORDS\n")
feature_result_fg = nfeature_accuracy_checker(ngram_range=(1, 4), stop_words=[''], n_features=n_features)

#feature_result_wosw = nfeature_accuracy_checker(stop_words=[''], n_features=n_features)

nfeatures_plot_ug = pd.DataFrame(feature_result_wosw,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_tg = pd.DataFrame(feature_result_tg,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_bg = pd.DataFrame(feature_result_bg,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_fg = pd.DataFrame(feature_result_fg,columns=['nfeatures','validation_accuracy','train_test_time'])
plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram')
plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy, label='trigram')
plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy, label='bigram')
plt.plot(nfeatures_plot_fg.nfeatures, nfeatures_plot_fg.validation_accuracy, label='fourgram')
plt.title("N-gram(1~4) test result : Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()
plt.show()


def train_test_and_evaluate(pipeline, x_train, y_train, x_test, y_test):
    if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
    sentiment_fit = pipeline.fit(x_train, y_train)
    y_pred = sentiment_fit.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0,1]))
    confusion = pd.DataFrame(conmat, index=['negative', 'positive'],
                         columns=['predicted_negative','predicted_positive'])
    print("null accuracy: {0:.2f}%".format(null_accuracy*100))
    print("accuracy score: {0:.2f}%".format(accuracy*100))
    if accuracy > null_accuracy:
        print("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100))
    elif accuracy == null_accuracy:
        print("model has the same accuracy with the null accuracy")
    else:
        print("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100))
    print("-"*80)
    print("Confusion Matrix\n")
    print(confusion)
    print("-"*80)
    print("Classification Report\n")
    print(classification_report(y_test, y_pred, target_names=['negative','positive']))

tg_cvec = CountVectorizer(max_features=200000,ngram_range=(1, 3))
tg_pipeline = Pipeline([
        ('vectorizer', tg_cvec),
        ('classifier', lr)
    ])
train_test_and_evaluate(tg_pipeline, x_train, y_train, x_validation, y_validation)





from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer

names = ["Logistic Regression", "Linear SVC", "LinearSVC with L1-based feature selection","Multinomial NB",
         "Bernoulli NB", "Ridge Classifier", "AdaBoost", "Perceptron","Passive-Aggresive", "Nearest Centroid"]
classifiers = [
    LogisticRegression(),
    LinearSVC(),
    Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
  ('classification', LinearSVC(penalty="l2"))]),
    MultinomialNB(),
    BernoulliNB(),
    RidgeClassifier(),
    AdaBoostClassifier(),
    Perceptron(),
    PassiveAggressiveClassifier(),
    NearestCentroid()
    ]
zipped_clf = zip(names,classifiers)

tvec = TfidfVectorizer()
def classifier_comparator(vectorizer=tvec, n_features=10000, stop_words=None, ngram_range=(1, 1), classifier=zipped_clf):
    result = []
    vectorizer.set_params(stop_words=stop_words, max_features=n_features, ngram_range=ngram_range)
    for n,c in classifier:
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', c)
        ])
        print("Validation result for {}".format(n))
        print(c)
        clf_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
        result.append((n,clf_accuracy,tt_time))
    return result

trigram_result = classifier_comparator(n_features=200000,ngram_range=(1,3))
