# -*- coding: utf-8 -*-
"""Test module for testing method methods_importer."""
import os

import pandas as pd
import pytest
from multi_label_classifier import MultiLabelClassifier
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split


class TestAtmIncClassifier:
    """Class for testing method methods_importer."""
    def initialization_test(self):
        """Test one module search."""
        clf = MultiLabelClassifier()
        assert isinstance(clf, MultiLabelClassifier)

    def test_method_not_found(self):
        """Test in case the module is not found."""
        df = pd.read_excel('ceo_train_v2.01.xlsx')
        df = df[['Комментарий', 'target']]
        target_cnt = df['target'].value_counts()
        targets_to_remove = target_cnt[target_cnt < 2].index
        df = df[~df['target'].isin(targets_to_remove)]
        model = MultiLabelClassifier()
        assert isinstance(model.fit(df['Комментарий'], df['target']), MultiLabelClassifier)

    def test_no_duplicates(self):
        """Test there are no duplicates in the output."""
        df = pd.read_excel(os.path.join('test_data', 'ceo_train_v2.01.xlsx'))
        df = df[['Комментарий', 'target']]
        target_cnt = df['target'].value_counts()
        targets_to_remove = target_cnt[target_cnt < 2].index
        df = df[~df['target'].isin(targets_to_remove)]

        X_train, X_test, y_train, y_test = train_test_split(
            df['Комментарий'], df['target'],
            test_size=0.2,
            random_state=42,
            stratify=df['target']
        )
        model = MultiLabelClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert jaccard_score(y_test, y_pred, average='samples').round(decimals=2) > 0.96

    def test_method_type_error(self):
        """Test raising TypeError."""
        with pytest.raises(TypeError):
            jaccard_score([1], [2])
