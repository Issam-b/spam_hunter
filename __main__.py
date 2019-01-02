from spam_hunter import SpamHunter
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import pandas as pd
from bs4 import BeautifulSoup
import requests
from textblob import TextBlob
from textblob import Word
import re
import nltk
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')


if __name__ == "__main__":
    # Use SpamHunter('Euron-spam', 40, 1, False) to process data again
    hunter = SpamHunter('Euron-spam', 40, 6, False)

    # # Naive Bayes based models
    [multinomial_nb_model, multi_nb_test_result] = hunter.train_model(MultinomialNB(), "MultinomialNB")
    [gaussian_nb_model, gaussian_nb_test_result] = hunter.train_model(GaussianNB(), "GaussianNB")
    [bernoulli_nb_model, bernoulli_nb_test_result] = hunter.train_model(BernoulliNB(), "BernoulliNB")
    
    # SVM based models
    [linear_svc_model, linear_test_result] = hunter.train_model(LinearSVC(max_iter=20000), "Linear SVC")
    [nusvc_model, nusvc_test_result] = hunter.train_model(NuSVC(gamma="auto"), "NuSVC")
    [svc_model, svc_test_result] = hunter.train_model(SVC(kernel="rbf", C=0.025, probability=True, gamma="auto"), "SVC")

    # Stochastic Gradient Descent based models
    [sgd_model, sgd_test_result] = hunter.train_model(SGDClassifier(loss="hinge", penalty="l2", max_iter=2000, tol=0.001, shuffle=True), "Stochastic Gradient Descent")

    # KNN based model
    [ncentroid_model, ncentroid_test_result] = hunter.train_model(NearestCentroid(), "Nearest Centroid")
    [knn_model, knn_test_result] = hunter.train_model(KNeighborsClassifier(n_neighbors=2), "KNearest Neighbor")

    # Decision Tree based
    [dtree_model, dtree_test_result] = hunter.train_model(tree.DecisionTreeClassifier(), "Decision Tree")

    # Gradient Tree Boosting
    [gboosting_model, gboosting_test_result] = hunter.train_model(GradientBoostingClassifier(n_estimators=100,
        learning_rate=1.0, max_depth=1, random_state=0), "Gradient Boosting")

    # Voting classifier
    [voting_model, voting_test_result] = hunter.train_model(VotingClassifier(estimators=[('dt', dtree_model), ('knn', knn_model),
        ('svc', svc_model)], voting='soft', weights=[2, 1, 2]), "Voting Classifier")

    # Neural Networks
    [nnet_model, nnet_test_result] = hunter.train_model(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), "Neural Networks")

    # Part 4
    nltk.download('averaged_perceptron_tagger')
    print ("PART 4")

    data = pd.read_csv("spam.csv",encoding='latin-1')
    data.shape
    data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    data.head()
    names = pd.read_csv("names.csv", delimiter='\t')
    names = pd.DataFrame(names.values, columns=['names'])
    names = names['names'].str.split(" ", n = 1, expand = True)
    names = names[0].tolist()
    
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
    def checkName(text):
        text = text.values[0]
        words = text.split(' ')
        count = 0
        for each in words:
            if each.upper() in names:
                count+=1
        return count
    
    def validURL(text):
        text = text.values[0]
        words = text.split(' ')
        count = 0
        for each in words:
            var = re.match(regex, each)
            if var is not None:
                if var == True:
                    count+=1
        return count

    def getURLNum(text):
        text = text.values[0]
        words = text.split(' ')
        count = 0
        for each in words:
            if len(each) > 3 and each[0] == 'h' and each[1] == 't' and each[2] =='t' and each[3] == 'p':
                count+=1
        return count

    def checkWord(text):
        text = text.values[0]
        words = text.split(' ')
        count = 0
        for each in words:
            w = Word(each)
            var = w.spellcheck()
            if var[0][1] <= 0.5:
                count+=1
        return count

    def countWords(text):
        text = text.values[0]
        words = text.split(' ')
        count = len(words)
        return count

    def pronouns(text):
        text = text.values[0]
        wiki = TextBlob(text)
        return len(wiki.tags)

    newList = []
    k = 0
    while k < data.shape[0]:
        newList.append(getURLNum(data['v2'][data['v2'].index==k]))
        k+=1
    data['numOfUrl'] = newList

    newList = []
    k = 0
    while k < data.shape[0]:
        newList.append(checkWord(data['v2'][data['v2'].index==k]))
        k+=1
    data['spellingMistakes'] = newList

    newList = []
    k = 0
    while k < data.shape[0]:
        newList.append(countWords(data['v2'][data['v2'].index==k]))
        k+=1
    data['countOfWords'] = newList

    newList = []
    k = 0
    while k < data.shape[0]:
        newList.append(checkName(data['v2'][data['v2'].index==k]))
        k+=1
    data['namesCount'] = newList

    newList = []
    k = 0
    while k < data.shape[0]:
        newList.append(validURL(data['v2'][data['v2'].index==k]))
        k+=1
    data['validURLCount'] = newList

    newList = []
    k = 0
    while k < data.shape[0]:
        newList.append(pronouns(data['v2'][data['v2'].index==k]))
        k+=1
    data['numProunous'] = newList

    data.pop('v2')
    label = data.pop('v1')
    print ("TRAINING WITH NEW FEATURES")
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=42)
    n = SVC()
    n.fit(X_train, y_train)
    prediction = n.predict(X_test)
    print ("SVM Accuracy ",accuracy_score(y_test, prediction))


    print ("PART 5")

    print ("TRAINING WITH TF.IDF FEATURES and PCA 10")
    fullData = pd.read_csv("spam.csv",encoding='latin-1')
    fullData = fullData.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

    corpus = []
    k = 0
    while k < fullData.shape[0]:
        corpus.append(fullData['v2'].values.tolist()[k])
        k+=1

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    X = X.toarray()

    n = PCA(10)
    pcaX = n.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(pcaX, label, test_size=0.33, random_state=42)

    n = SVC()
    n.fit(X_train, y_train)
    prediction = n.predict(X_test)
    print ("SVM Accuracy ",accuracy_score(y_test, prediction))