import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import nltk
import data_analysis_6


class FeatEngr:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer()

    def build_train_features(self, examples):
        """
        Method to take in training text features and do further feature engineering
        Most of the work in this homework will go here, or in similar functions
        :param examples: currently just a list of forum posts
        """
        from sklearn.feature_extraction.text import CountVectorizer
        import scipy as sp
        from scipy.sparse import csr_matrix

        trope_vectorizer = CountVectorizer()
        page_vectorizer = CountVectorizer()
        feature_5 = data_analysis_6.give_genre_vector(list(examples["page"]))
        feature_1 = self.vectorizer.fit_transform(list(examples["sentence"]))
        feature_2 = trope_vectorizer.fit_transform(list(examples["trope"]))
        feature_3 = page_vectorizer.fit_transform(list(examples["page"]))
        feature_4 = [len(x) for x in list(examples["sentence"])]
        training_vec = sp.sparse.hstack((feature_1, feature_2, feature_3, csr_matrix(feature_4).T))
        return training_vec

    def get_test_features(self, examples):
        """
        Method to take in test text features and transform the same way as train features
        :param examples: currently just a list of forum posts
        """
        return self.vectorizer.transform(examples)

    def show_top10(self):
        """
        prints the top 10 features for the positive class and the
        top 10 features for the negative class.
        """
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        top10 = np.argsort(self.logreg.coef_[0])[-10:]
        bottom10 = np.argsort(self.logreg.coef_[0])[:10]
        print("Pos: %s" % " ".join(feature_names[top10]))
        print("Neg: %s" % " ".join(feature_names[bottom10]))

    def train_model(self, random_state=1234):
        """
        Method to read in training data from file, and
        train Logistic Regression classifier.

        :param random_state: seed for random number generator
        """

        from sklearn.linear_model import LogisticRegression

        # load data
        dfTrain = pd.read_csv("train.csv")

        # get training features and labels
        self.X_train = self.build_train_features(dfTrain)
        self.y_train = np.array(dfTrain["spoiler"], dtype=int)

        # train logistic regression model.  !!You MAY NOT CHANGE THIS!!
        self.logreg = LogisticRegression(random_state=random_state)
        self.logreg.fit(self.X_train, self.y_train)

    def train_test_validation_model(self, random_state=1234):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import KFold
        from scipy.sparse import csr_matrix

        # load data
        dfTrain = pd.read_csv("train.csv")

        # get training features and labels
        self.X_train = self.build_train_features(dfTrain)
        self.y_train = np.array(dfTrain["spoiler"], dtype=int)

        # train logistic regression model.  !!You MAY NOT CHANGE THIS!!
        accuracy = []
        kf = KFold(n_splits=10, random_state=None, shuffle=True)
        for train_index, test_index in kf.split(self.X_train):
            x_train, x_test = csr_matrix(self.X_train)[train_index], csr_matrix(self.X_train)[test_index]
            y_train, y_test = self.y_train[train_index], self.y_train[test_index]
            self.logreg = LogisticRegression(random_state=random_state)
            self.logreg.fit(x_train, y_train)
            y_pred = self.logreg.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            accuracy.append(acc)
        print(sum(accuracy)/len(accuracy))




    def model_predict(self):
        """
        Method to read in test data from file, make predictions
        using trained model, and dump results to file
        """

        # read in test data
        dfTest = pd.read_csv("test.csv")

        # featurize test data
        self.X_test = self.get_test_features(list(dfTest["sentence"]))

        # make predictions on test data
        pred = self.logreg.predict(self.X_test)

        # dump predictions to file for submission to Kaggle
        pd.DataFrame({"spoiler": np.array(pred, dtype=bool)}).to_csv("prediction.csv", index=True, index_label="Id")


# Instantiate the FeatEngr clas
feat = FeatEngr()

# Train your Logistic Regression classifier
feat.train_test_validation_model(random_state=1230)

# Shows the top 10 features for each class
#feat.show_top10()

