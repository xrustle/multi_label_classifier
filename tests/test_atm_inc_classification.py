# -*- coding: utf-8 -*-
"""Test module for testing method methods_importer."""
import os

import pandas as pd
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split

from multi_label_classifier import MultiLabelClassifier


class TestAtmIncClassifier:
    """Class for testing method methods_importer."""

    def test_initialization(self):
        """Test one module search."""
        clf = MultiLabelClassifier()
        assert isinstance(clf, MultiLabelClassifier)

    def test_fitting(self):
        """Test in case the module is not found."""
        test_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'test_data',
            'ceo_train_v2.01.xlsx',
        )
        df = pd.read_excel(test_data_path)
        df = df[['Комментарий', 'target']]
        target_cnt = df['target'].value_counts()
        targets_to_remove = target_cnt[target_cnt < 2].index
        df = df[~df['target'].isin(targets_to_remove)]
        model = MultiLabelClassifier(n_estimators=1)
        assert isinstance(
            model.fit(df['Комментарий'], df['target'], silent=False),
            MultiLabelClassifier,
        )

    def test_prediction(self):
        """Test there are no duplicates in the output."""
        df = pd.read_excel(os.path.join('test_data', 'ceo_train_v2.01.xlsx'))
        df = df[['Комментарий', 'target']]
        target_cnt = df['target'].value_counts()
        targets_to_remove = target_cnt[target_cnt < 2].index
        df = df[~df['target'].isin(targets_to_remove)]

        X_train, X_test, y_train, y_test = train_test_split(
            df['Комментарий'],
            df['target'],
            test_size=0.2,
            random_state=42,
            stratify=df['target'],
        )
        model = MultiLabelClassifier(n_estimators=10,)
        model.fit(X_train, y_train, silent=False)
        y_pred = model.predict(X_test)
        y_test = model.mlb.transform(y_test.apply(lambda x: x.split(',')))
        print(jaccard_score(y_test, y_pred, average='samples').round(decimals=3))
        assert (
            jaccard_score(y_test, y_pred, average='samples').round(decimals=2) >= 0.96
        )
