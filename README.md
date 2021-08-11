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
Пример установки и использования при решении задачи классификации инцидентов банкоматов в ноутбуке по ссылке ниже

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/...)
