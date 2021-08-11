# -*- coding: utf-8 -*-
"""Module consists of the only one function methods_importer."""
import os
import re
import warnings

import nltk
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import jaccard_score
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


class MultiLabelClassifier:
    """Multi label classifier model based on XGBoost chain."""

    def __init__(
        self,
        n_estimators=10,
        var_threshold=5e-5,
        learning_rate=0.5,
        explained_variance_ratio=0.99,
    ):
        """
        Create a new classifier model instance.

        Args:
            -n_estimators- number of ChainClassifiers
            -silent- disables messages during training
        """
        self.n_estimators = n_estimators
        self.explained_variance_ratio = explained_variance_ratio
        param_dist = dict(
            learning_rate=learning_rate,
            objective='binary:logistic',
            use_label_encoder=False,
            verbosity=0,
            gpu_id=0,
            tree_method='gpu_hist',
            predictor='cpu_predictor',
        )
        self.base_clf = XGBClassifier(**param_dist)
        self.chains = [
            ClassifierChain(self.base_clf, order='random', random_state=i)
            for i in range(self.n_estimators)
        ]
        self.mlb = MultiLabelBinarizer()
        self.vect = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.5,
            preprocessor=self.doc_preprocessing,
        )
        self.constant_filter = VarianceThreshold(threshold=var_threshold)
        self.pca = PCA(svd_solver='full')
        stem_filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'stop_stems.txt'
        )
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_ru')
        except LookupError:
            nltk.download('averaged_perceptron_tagger_ru')
        with open(stem_filename, 'r', encoding='utf-8') as f:
            self.stop_stems = f.read().splitlines()

    def doc_preprocessing(self, string):
        """
        Preprocess one doc from corpus.

        Args:
            -string- document

        Returns:
            output- filtered string with accepted stems
        """
        string = string.lower()
        string = str.replace(string, 'ё', 'е')
        string = re.sub(r'\?+', ' вопрос ', string)
        prog = re.compile('[а-я]+')
        words = prog.findall(string.lower())
        stopwords = nltk.corpus.stopwords.words('russian')
        words = [w for w in words if w not in stopwords or w in ['не', 'нет']]
        functionalPos = {'PRCL', 'CONJ'}
        words = [
            w for w, pos in nltk.pos_tag(words, lang='rus') if pos not in functionalPos
        ]
        stemmer = SnowballStemmer('russian')
        stems = [s for s in map(stemmer.stem, words) if s not in self.stop_stems]
        return ' '.join(stems)

    def docs_to_vec(self, X):
        """
        Create a new classifier model instance.

        Args:
            -X- design matrix or corpus

        Returns:
            output- vectors dataframe
        """
        X = self.vect.transform(X)
        feature_names = self.vect.get_feature_names()
        df = pd.DataFrame(X.toarray(), columns=feature_names)
        return df

    def fit(self, X, y, silent=True):
        """
        Train model with design matrix X and labels y.

        Args:
            -X- design matrix
            -y- comma-separated labels
            -silent- disables messages during training

        Returns:
            output- self-instance for further use
        """
        if not silent:
            print('Data preprocessing...')
        one_hot_labels = self.mlb.fit_transform(y.apply(lambda x: x.split(',')))
        self.vect.fit(X)
        x_train = self.docs_to_vec(X)
        self.constant_filter.fit(x_train)
        x_train = self.constant_filter.transform(x_train)
        self.pca.fit(x_train)
        n_features = np.argmax(
            self.pca.explained_variance_ratio_.cumsum() > self.explained_variance_ratio
        )
        self.pca = PCA(n_components=n_features, svd_solver='full')
        x_train = self.pca.fit_transform(x_train)
        if not silent:
            print('Training...')
        if silent:
            for chain in self.chains:
                chain.fit(x_train, one_hot_labels)
        else:
            for chain in tqdm(self.chains):
                chain.fit(x_train, one_hot_labels)
        return self

    def predict_proba(self, X):
        """
        Predict the probability of belonging to each class.

        Args:
            -X- design matrix

        Returns:
            output- array of probabilities with shape number of samples times number
            of classes
        """
        x_test = self.docs_to_vec(X)
        x_test = self.constant_filter.transform(x_test)
        x_test = self.pca.transform(x_test)
        y_pred = np.array([chain.predict_proba(x_test) for chain in self.chains])
        return y_pred.mean(axis=0)

    def predict(self, X, threshold=0.5):
        """
        Predict one-hot-encoded labels.

        Args:
            -X- design matrix
            -threshold-

        Returns:
            output- array of zeros and ones with shape number of samples times
            number of classes
        """
        y_pred = self.predict_proba(X) >= threshold
        if 'empty' in self.mlb.classes_:
            empty = self.mlb.transform([['empty']])[0] == 1
            y_pred_sum = y_pred.sum(axis=1)
            y_pred[y_pred_sum == 0] = empty
        return y_pred

    def predict_labels(self, X, threshold=0.5):
        """
        Predict comma separated text labels.

        Args:
            -X- design matrix
            -threshold-

        Returns:
            output- array of comma separated lists of classes names
        """
        y_pred = self.predict(X, threshold)
        return np.array(list(map(','.join, self.mlb.inverse_transform(y_pred))))

    def score(self, X, y):
        """
        Return Jaccard score.

        Args:
            -X- design matrix
            -y- true labels

        Returns:
            output- Jaccard score
        """
        y_true = self.mlb.transform(y.apply(lambda x: x.split(',')))
        return jaccard_score(y_true, self.predict(X))
