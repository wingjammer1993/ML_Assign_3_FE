import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import nltk
from nltk.stem import *
from nltk.stem.porter import *
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import data_analysis_9
import data_analysis_8


def stem_pos_sentences(examples):
    new_examples = []
    new_pos = []
    stemmer = PorterStemmer()
    lemmer = WordNetLemmatizer()
    for ex in examples:
        gen_list = word_tokenize(ex)
        pos = nltk.pos_tag(gen_list)
        pos = [p[-1] for p in pos if p[-1]]
        lemms = [lemmer.lemmatize(ex) for ex in gen_list]
        singles = [stemmer.stem(ex) for ex in lemms]
        new_examples.append(' '.join(singles))
        new_pos.append(' '.join(pos))
    return new_examples, new_pos


def stem_pos_tropes(examples):
    new_examples = []
    stemmer = PorterStemmer()
    lemmer = WordNetLemmatizer()
    for ex in examples:
        ex = re.sub(r"\B([A-Z])", r" \1", ex)
        gen_list = word_tokenize(ex)
        lemms = [lemmer.lemmatize(ex) for ex in gen_list]
        singles = [stemmer.stem(ex) for ex in lemms]
        new_examples.append(' '.join(singles))
    return new_examples



class FeatEngr:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_extraction.text import CountVectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words={'English'})
        self.page_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words={'English'})
        self.tag_vectorizer = CountVectorizer()


    def build_train_features(self, examples):
        """
        Method to take in training text features and do further feature engineering
        Most of the work in this homework will go here, or in similar functions
        :param examples: currently just a list of forum posts
        """

        import scipy as sp
        from scipy.sparse import csr_matrix

        stemmed_sentences, tags = stem_pos_sentences(list(examples["sentence"]))
        stemmed_pages = stem_pos_tropes(list(examples["trope"]))
        feature_1 = self.vectorizer.fit_transform(stemmed_sentences)
        feature_2 = self.tag_vectorizer.fit_transform(tags)
        feature_3 = self.page_vectorizer.fit_transform(stemmed_pages)
        training_vec = sp.sparse.hstack((feature_1, feature_2, feature_3))
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
        feature_names = np.asarray(self.tag_vectorizer.get_feature_names())
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
        from sklearn.metrics import confusion_matrix

        # load data
        dfTrain = pd.read_csv("train.csv")
        sentences = list(dfTrain["sentence"])

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
            for index, i in enumerate(y_test):
                if y_test[index] != y_pred[index]:
                    required_index = test_index[index]
                    print(sentences[required_index])
                    print(y_test[index])
                    print(y_pred[index])

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

