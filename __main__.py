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
    hunter = SpamHunter('Euron-spam', 40, 6, False, use_extra_features=False)

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

    # F1 F2 F3 F4 F5 F6
    hunter = SpamHunter('Euron-spam', 40, 6, False, use_extra_features=True)

    [multinomial_nb_model, multi_nb_test_result] = hunter.train_model(MultinomialNB(), "MultinomialNB")
    [linear_svc_model, linear_test_result] = hunter.train_model(LinearSVC(max_iter=20000), "Linear SVC")
    [sgd_model, sgd_test_result] = hunter.train_model(SGDClassifier(loss="hinge", penalty="l2", max_iter=2000, tol=0.001, shuffle=True), "Stochastic Gradient Descent")
    [dtree_model, dtree_test_result] = hunter.train_model(tree.DecisionTreeClassifier(), "Decision Tree")
    [nnet_model, nnet_test_result] = hunter.train_model(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), "Neural Networks")