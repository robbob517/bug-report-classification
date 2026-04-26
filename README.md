# Performance-related Bug Report Classification

## Requirements

- Python 3.12+

- Required Packages:

``` python
pandas
numpy
scikit-learn
nltk
sentence-transformers
```

If CUDA is available

``` python
pip uninstall torch
```

then

``` python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## How to Run

Run one at a time in order below, waiting for output to be complete before running next line

``` python
python tfidf_nb.py
```

``` python
python sbert_svm.py
```

``` python
python wilcoxon.py
```
