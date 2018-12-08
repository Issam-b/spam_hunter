import os
from sys import stdout
import math
from collections import Counter
import pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class SpamHunter(object):

    def __init__(self, train_dir, test_ratio, slices_count, debug=False):
        self.debug = debug
        self.dict_size = 3000
        [self.emails_size, self.emails_train, self.emails_test, self.train_labels, self.test_labels] = self.process_files(train_dir, test_ratio, slices_count)
        print("Creating dictionary")
        self.dictionary = self.make_Dictionary(self.emails_train)
        print("Extracting features for training")
        self.train_matrix = self.extract_features(self.emails_train)
        print("Extracting features for testing")
        self.test_matrix = self.extract_features(self.emails_test)

    def make_Dictionary(self, emails):
        all_words = []
        for email in emails:
            try:
                with open(email, encoding="utf8", errors='ignore') as m:
                    for i, line in enumerate(m):
                        words = line.split()
                        all_words += words
            except:
                if(self.debug):
                    print("Reading error in file: " + str(email))
            
        self.dictionary = Counter(all_words)

        list_to_remove = self.dictionary.keys()
        for item in list(list_to_remove):
            if item.isalpha() == False: 
                del self.dictionary[item]
            elif len(item) == 1:
                del self.dictionary[item]
        self.dictionary = self.dictionary.most_common(self.dict_size)

        return self.dictionary

    def extract_features(self, emails, use_idf=True):
        features_matrix = np.zeros((len(emails), self.dict_size))
        docID = 0
        df_matrix = np.zeros(self.dict_size)
        docs_count = len(emails)
        for email in emails:
            stdout.write("\rExtracting features %d/%d files" % (docID, len(emails)))
            stdout.flush()
            try:
                with open(email, encoding="utf8", errors='ignore') as m:
                    for i, line in enumerate(m):
                        words = line.split()
                        for word in words:
                            if(word.isalpha() == False or len(word) == 1):
                                break
                            for i, dict_word in enumerate(self.dictionary):
                                if dict_word[0] == word:
                                    features_matrix[docID, i] = words.count(word)
                                    break
                    # Compute document frequency df values
                    for i in range(0, self.dict_size):
                        if(features_matrix[docID, i] > 0):
                            df_matrix[i] += 1
                    docID += 1
            except:
                if(self.debug):
                    print("\nReading error in file: " + str(email))

        if(use_idf):
            self.compute_tf_idf(features_matrix, df_matrix, docs_count)

        print("\n")
        return features_matrix

    def compute_tf_idf(self, features_matrix, df_matrix, docs_count):
        print("\n")
        for doc_id in range(0, docs_count):
            stdout.write("\rComputing TF-IDF %d/%d docs" % (doc_id, docs_count))
            stdout.flush()
            for i in range(0, self.dict_size):
                if(features_matrix[doc_id, i] != 0):
                    features_matrix[doc_id, i] *= math.log(docs_count / df_matrix[i])

    def train_gaussian_nb(self):
        self.model_gaussian_nb = GaussianNB()
        print("GaussianNB training")
        self.model_gaussian_nb.fit(self.train_matrix, self.train_labels)

        print("Test with GaussianNB")
        test_result = self.test_model(self.model_gaussian_nb, self.test_matrix, self.test_labels)

        return self.model_gaussian_nb, test_result

    def train_nusvc(self):
        self.model_nusvc = NuSVC(gamma="auto")
        print("NuSVC training")
        self.model_nusvc.fit(self.train_matrix, self.train_labels)

        print("Test with NuSVC")
        test_result = self.test_model(self.model_nusvc, self.test_matrix, self.test_labels)

        return self.model_nusvc, test_result

    def train_svc(self):
        self.model_svc = SVC(kernel="rbf", C=0.025, probability=True, gamma="auto")
        print("SVC training")
        self.model_svc.fit(self.train_matrix, self.train_labels)

        print("Test with SVC")
        test_result = self.test_model(self.model_svc, self.test_matrix, self.test_labels)

        return self.model_svc, test_result

    def train_multinomial_nb(self):
        self.model_multinomial_nb = MultinomialNB()
        print("MultinomialNB training")
        self.model_multinomial_nb.fit(self.train_matrix, self.train_labels)

        print("Test with MultinomialNB")
        test_result = self.test_model(self.model_multinomial_nb, self.test_matrix, self.test_labels)

        return self.model_multinomial_nb, test_result

    def train_linear_svc(self):
        self.model_linear_svc = LinearSVC(max_iter=20000)
        print("Linear SVC training")
        self.model_linear_svc.fit(self.train_matrix, self.train_labels)

        print("Test with Linear SVC")
        test_result = self.test_model(self.model_linear_svc, self.test_matrix, self.test_labels)

        return self.model_linear_svc, test_result

    def test_model(self, model, test_matrix, test_labels):
        result = model.predict(test_matrix)
        print("Accuracy: " + str(accuracy_score(test_labels, result)) + "\n")
        print("Confusion matrix: " + str(confusion_matrix(test_labels, result)))
        return result

    def print_dictionary_line_by_line(self, dictionary):
        for item in dictionary:
            print (item)

    def process_files(self, mail_root, test_ratio, slices_count):
        [emails_spam, emails_ham] = self.get_list_files_euron(mail_root, slices_count)

        emails_size = len(emails_ham + emails_spam)
        spam_size = len(emails_spam)
        ham_size = len(emails_ham)
        print("Total emails: " + str(emails_size) + " - Hams: " + str(ham_size) + " - Spams: " + str(spam_size))
        spam_split = round(spam_size * (100 - test_ratio) / 100)
        ham_split = round(ham_size * (100 - test_ratio) / 100)
        train_spam_emails = emails_spam[:spam_split]
        train_ham_emails = emails_ham[:ham_split]
        emails_train = train_spam_emails + train_ham_emails
        train_labels = np.zeros(len(emails_train))
        train_labels[:len(train_spam_emails)] = 1

        test_spam_emails = emails_spam[spam_split:]
        test_ham_emails = emails_ham[ham_split:]
        emails_test = test_spam_emails + test_ham_emails
        test_labels = np.zeros(len(emails_test))
        test_labels[:len(test_spam_emails)] = 1

        return emails_size, emails_train, emails_test, train_labels, test_labels

    def get_list_files_euron(self, mail_root, slices_count):
        emails_ham = []
        emails_spam = []
        for i in range(1, slices_count + 1):
            ham_path = mail_root + "/enron" + str(i) + "/ham"
            spam_path = mail_root + "/enron" + str(i) + "/spam"
            emails_ham += [os.path.join(ham_path, file_in) for file_in in os.listdir(ham_path)]
            emails_spam += [os.path.join(spam_path, file_in) for file_in in os.listdir(spam_path)]

        return emails_spam, emails_ham

    def get_directory_files_count(self, dir):
        return len([1 for x in list(os.scandir(dir)) if x.is_file()])
