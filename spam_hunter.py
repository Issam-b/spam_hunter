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

    def __init__(self, train_dir, test_ratio, slices_count, use_processed=True, debug=False):
        """
        Init for SpamHunter class, it calls methods to create words dictionary,
        create train and test datasets and extract their features.

        @param train_dir Root directory of Euron-spam folders
        @param test_ration Ratio between train and test datasets for all input emails
        @param slices_count The number of sub-directories to consider from Euron-spam dataset
        @param debug print debug output such as exception messages
        """
        if(use_processed == False):
            self.debug = debug
            self.dict_size = 3000
            [self.emails_train, self.emails_test, self.train_labels, self.test_labels] = self.process_files(train_dir, test_ratio, slices_count)
            print("Creating dictionary")
            self.dictionary = self.make_Dictionary(self.emails_train)
            print("Extracting features for training")
            self.train_matrix = self.extract_features(self.emails_train)
            print("Extracting features for testing")
            self.test_matrix = self.extract_features(self.emails_test)
            self.save_processed_datasets("processed_datasets")
        else:
            self.load_processed_datasets("processed_datasets")

    def save_processed_datasets(self, file_name):
        """
        Save data processed (feature matrices and dictionary) to be used directly next time

        @param file_name Name of saved file
        """
        print("Saving processed data to file" + str(file_name) + ".npz")
        np.savez_compressed(str(file_name) + ".npz", dictionary=self.dictionary, train_matrix=self.train_matrix, 
            test_matrix=self.test_matrix, train_labels=self.train_labels, test_labels=self.test_labels)

    def load_processed_datasets(self, file_name):
        """
        Load data processed (feature matrices and dictionary) to be used directly without need to process again

        @param file_name Name of file to load
        """
        print("Loading processed data from file" + str(file_name) + ".npz")
        processed = np.load(str(file_name) + ".npz")
        self.dictionary = processed["dictionary"]
        self.train_matrix = processed["train_matrix"]
        self.test_matrix = processed["test_matrix"]
        self.train_labels = processed["train_labels"]
        self.test_labels = processed["test_labels"] 

    def make_Dictionary(self, emails):
        """
        Create a word dictionary from list of email paths
        
        @param emails Email paths list
        """
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
        """
        Extract features matrix from emails according to the word dictionary we have

        @param emails List of email paths
        @param use_idf if True, use IF-IDF as features not only word frequency
        """
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
        """
        Compute the TF-IDF matrix of features, called by extract_features()

        @param features_matrix Matrix to update with TF-IDF
        @param df_matrix The document frequency df matrix
        @param docs_count number of docs to consider
        """
        print("\n")
        for doc_id in range(0, docs_count):
            stdout.write("\rComputing TF-IDF %d/%d docs" % (doc_id, docs_count))
            stdout.flush()
            for i in range(0, self.dict_size):
                if(features_matrix[doc_id, i] != 0):
                    features_matrix[doc_id, i] *= math.log(docs_count / df_matrix[i])

    def train_gaussian_nb(self):
        """
        Train spam hunter using Gaussian NB classifier
        """
        self.model_gaussian_nb = GaussianNB()
        test_result = []
        print("GaussianNB training")
        try:
            self.model_gaussian_nb.fit(self.train_matrix, self.train_labels)
            print("Test with GaussianNB")
            test_result = self.test_model(self.model_gaussian_nb, self.test_matrix, self.test_labels)
        except Exception as e:
            print("Training error: " + str(e))

        return self.model_gaussian_nb, test_result

    def train_nusvc(self):
        """
        Train spam hunter using NuSVC classifier
        """
        self.model_nusvc = NuSVC(gamma="auto")
        test_result = []
        print("NuSVC training")
        try:    
            self.model_nusvc.fit(self.train_matrix, self.train_labels)
            print("Test with NuSVC")
            test_result = self.test_model(self.model_nusvc, self.test_matrix, self.test_labels)
        except Exception as e:
            print("Training error: " + str(e))

        return self.model_nusvc, test_result

    def train_svc(self):
        """
        Train spam hunter using SVC classifier
        """
        self.model_svc = SVC(kernel="rbf", C=0.025, probability=True, gamma="auto")
        test_result = []
        print("SVC training")
        try:
            self.model_svc.fit(self.train_matrix, self.train_labels)
            print("Test with SVC")
            test_result = self.test_model(self.model_svc, self.test_matrix, self.test_labels)
        except Exception as e:
            print("Training error: " + str(e))

        return self.model_svc, test_result

    def train_multinomial_nb(self):
        """
        Train spam hunter using Multinomial NB classifier
        """
        self.model_multinomial_nb = MultinomialNB()
        print("MultinomialNB training")
        self.model_multinomial_nb.fit(self.train_matrix, self.train_labels)

        print("Test with MultinomialNB")
        test_result = self.test_model(self.model_multinomial_nb, self.test_matrix, self.test_labels)

        return self.model_multinomial_nb, test_result

    def train_linear_svc(self):
        """
        Train spam hunter using Linear SCV classifier
        """
        self.model_linear_svc = LinearSVC(max_iter=20000)
        print("Linear SVC training")
        self.model_linear_svc.fit(self.train_matrix, self.train_labels)

        print("Test with Linear SVC")
        test_result = self.test_model(self.model_linear_svc, self.test_matrix, self.test_labels)

        return self.model_linear_svc, test_result

    def test_model(self, model, test_matrix, test_labels):
        """
        Test a trained model for a given test set, prints accuracy and confusion matrix.
        Returns the predicted matrix for test emails dataset

        @param model Training model used
        @param test_matrix Features matrix for the test dataset
        @parm test_labels Test dataset labels
        """
        result = model.predict(test_matrix)
        print("Accuracy: " + str(accuracy_score(test_labels, result)))
        print("Confusion matrix: " + str(confusion_matrix(test_labels, result)) + "\n")
        return result

    def process_files(self, mail_root, test_ratio, slices_count):
        """
        Process spam dataset root dir to list and divide train and test sets

        @param mail_root Root dir for spam folders
        @param test_ratio Ratio with which with to divide train and test sets
        @param slices_count Number of part of set to consider
        """
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

        return emails_train, emails_test, train_labels, test_labels

    def get_list_files_euron(self, mail_root, slices_count):
        """
        Get list of emails files from Euron-spam dataset, divided to spam and ham sets

        @param mail_root Root directory of Euron-spam dataset
        @param slices_count Number of part of set to consider
        """
        emails_ham = []
        emails_spam = []
        for i in range(1, slices_count + 1):
            ham_path = mail_root + "/enron" + str(i) + "/ham"
            spam_path = mail_root + "/enron" + str(i) + "/spam"
            emails_ham += [os.path.join(ham_path, file_in) for file_in in os.listdir(ham_path)]
            emails_spam += [os.path.join(spam_path, file_in) for file_in in os.listdir(spam_path)]

        return emails_spam, emails_ham
