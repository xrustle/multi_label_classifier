# -*- coding: utf-8 -*-
"""Module consists of the only one function methods_importer."""
import re

import nltk
import numpy as np
import pandas as pd
# import warnings
from nltk.stem.snowball import SnowballStemmer
from tqdm.autonotebook import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import jaccard_score
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier

# warnings.filterwarnings('ignore')


class MultiLabelClassifier:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        param_dist = dict(objective='binary:logistic', use_label_encoder=False, verbosity=0)
        self.base_clf = XGBClassifier(**param_dist)
        self.chains = [ClassifierChain(self.base_clf, order='random', random_state=i)
                       for i in range(self.n_estimators)]
        self.mlb = MultiLabelBinarizer()
        self.vect = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.5,
            preprocessor=self.doc_preprocessing
        )
        self.constant_filter = VarianceThreshold(threshold=5e-5)
        with open('stop_stems.txt', 'r', encoding='utf-8') as f:
            self.stop_stems = f.read().splitlines()

    def doc_preprocessing(self, string):
        string = string.lower()
        string = str.replace(string, 'ё', 'е')
        string = re.sub(r'\?+', ' вопрос ', string)
        prog = re.compile('[а-я]+')
        words = prog.findall(string.lower())
        stopwords = nltk.corpus.stopwords.words('russian')
        words = [w for w in words if w not in stopwords or w in ['не', 'нет']]
        functionalPos = {'PRCL', 'CONJ'}
        words = [w for w, pos in nltk.pos_tag(words, lang='rus') if pos not in functionalPos]
        stemmer = SnowballStemmer('russian')
        stems = [s for s in map(stemmer.stem, words) if s not in self.stop_stems]
        return ' '.join(stems)

    def docs_to_vec(self, X):
        X = self.vect.transform(X)
        feature_names = self.vect.get_feature_names()
        df = pd.DataFrame(X.toarray(), columns=feature_names)
        return df

    def fit(self, X, y):
        one_hot_labels = self.mlb.fit_transform(y.apply(lambda x: x.split(',')))
        X = self.vect.fit(X.values)
        X = self.docs_to_vec(X)
        X = self.constant_filter.fit_transform(X)
        for chain in tqdm(self.chains):
            chain.fit(X, one_hot_labels)
        return self

    def predict_proba(self, X):
        X = self.docs_to_vec(X)
        X = self.constant_filter.transform(X)
        y_pred = np.array([chain.predict(X) for chain in self.chains])
        return y_pred.mean(axis=0)

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def predict_labels(self, X, threshold=0.5):
        y_pred = self.predict(X, threshold)
        return np.array(map(''.join, self.mlb.inverse_transform(y_pred)))

    def score(self, X, y):
        return jaccard_score(y, self.predict(X))
