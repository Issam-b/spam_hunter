import os
from sys import stdout
import math
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

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
        print("Saving processed data to file " + str(file_name) + ".npz")
        np.savez_compressed(str(file_name) + ".npz", dictionary=self.dictionary, train_matrix=self.train_matrix, 
            test_matrix=self.test_matrix, train_labels=self.train_labels, test_labels=self.test_labels)

    def load_processed_datasets(self, file_name):
        """
        Load data processed (feature matrices and dictionary) to be used directly without need to process again

        @param file_name Name of file to load
        """
        print("Loading processed data from file " + str(file_name) + ".npz")
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
                        if len(line) > 0: # starting from line 2
                            words = line.split()
                            if i == 0:
                                words.pop(0)
                            all_words += words
            except:
                if(self.debug):
                    print("Reading error in file: " + str(email))
        
        stop_words = stopwords.words('english')
        all_words = [word for word in all_words if word not in stop_words and word.isalpha() and len(word) > 1]

        lemmatizer = WordNetLemmatizer()
        all_words = [lemmatizer.lemmatize(word) for word in all_words]
        self.dictionary = Counter(all_words)

        self.dictionary = self.dictionary.most_common(self.dict_size)

        return self.dictionary

    def extract_features(self, emails, use_idf=True):
        """
        Extract features matrix from emails according to the word dictionary we have

        @param emails List of email paths
        @param use_idf if True, use IF-IDF as features not only word frequency
        """
        features_matrix = np.zeros((len(emails), self.dict_size + 2))
        docID = 0
        df_matrix = np.zeros(self.dict_size)
        docs_count = len(emails)
        url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

        stop_words = stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        for email in emails:
            num_words = 0
            stdout.write("\rExtracting features %d/%d files" % (docID, len(emails)))
            stdout.flush()
            try:
                with open(email, encoding="utf8", errors='ignore') as m:
                    for i, line in enumerate(m):
                        if len(line) > 0: # starting from line 2
                            words = line.split()
                            if i == 0:
                                words.pop(0)
                            num_words += len(words)
                            words = [word for word in words if word not in stop_words and word.isalpha() and len(word) > 1]
                            words = [lemmatizer.lemmatize(word) for word in words]

                            for word in words:
                                for j, dict_word in enumerate(self.dictionary):
                                    if dict_word[0] == word:
                                        features_matrix[docID, j] = words.count(word)
                                        break

                            # Count URLs number in current line
                            urls = re.findall(url_regex, line)
                            if len(urls) > 0:
                                features_matrix[docID, self.dict_size] += len(urls)
                            
                            # Number of words in line
                            features_matrix[docID, self.dict_size + 1] += num_words

                    # Compute document frequency df values
                    for x in range(0, self.dict_size):
                        if(features_matrix[docID, x] > 0):
                            df_matrix[x] += 1
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

    def train_model(self, model, model_name):
        """
        Train spam hunter using given model
        """
        print(str(model_name) + " training")
        model.fit(self.train_matrix, self.train_labels)

        print("Test with " + str(model_name))
        test_result = self.test_model(model, self.test_matrix, self.test_labels)

        return model, test_result

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
        spam_split = int(round(spam_size * (100 - test_ratio) / 100))
        ham_split = int(round(ham_size * (100 - test_ratio) / 100))
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
