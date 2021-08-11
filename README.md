# Multi label Classifier

Класс `MultiLabelClassifier` предназначен для классификации коротких текстов, которые могут одновременно принадлежать нескольким классам.

## Installing from source

```pip install git+https://github.com/xrustle/multi_label_classifier```

## Usage example
Стиль именования методов подобен `sklearn`
```python
from multi_label_classifier import MultiLabelClassifier

model = MultiLabelClassifier()
model.fit(X, y)
model.predict(X_test)
```
Пример установки и использования на данных инцидентов на банкоматах:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/girafe-ai/ml-mipt/blob/msai_ml_s21/week02_linear_regression/week02_Linear_regression_and_SGD.ipynb)
