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
    hunter = SpamHunter('Euron-spam', 40, 6, False) #6

    # Part 4
    [multinomial_nb_model, multi_nb_test_result] = hunter.train_model(MultinomialNB(), "MultinomialNB")
    [linear_svc_model, linear_test_result] = hunter.train_model(LinearSVC(max_iter=20000), "Linear SVC")
    [sgd_model, sgd_test_result] = hunter.train_model(SGDClassifier(loss="hinge", penalty="l2", max_iter=2000, tol=0.001, shuffle=True), "Stochastic Gradient Descent")
    [dtree_model, dtree_test_result] = hunter.train_model(tree.DecisionTreeClassifier(), "Decision Tree")
    [nnet_model, nnet_test_result] = hunter.train_model(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), "Neural Networks")


