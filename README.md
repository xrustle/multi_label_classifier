# Multi label Classifier

Класс `MultiLabelClassifier` предназначен для классификации коротких текстов, которые могу тодновременно принадлежать нескольким классам. 

## Installing from source

```pip install git+https://github.com/xrustle/multi_label_classifier```

## Пример использования

```python
from multi_label_classifier import MultiLabelClassifier

model = MultiLabelClassifier()
model.fit(X, y)
model.predict(X_test)
```

## Testing

- ```make install-dev```
- ```make test```
