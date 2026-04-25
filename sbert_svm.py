########## 1. Import required libraries ##########

import pandas as pd
import numpy as np
import re

# Sentence-BERT (Embeddings)
from sentence_transformers import SentenceTransformer

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)

# SVM (Classifier)
from sklearn.svm import SVC

# Text cleaning & stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

########## 2. Define text preprocessing methods ##########

def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Stopwords
NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']  # You can customize this list as needed
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

def remove_stopwords(text):
    """Remove stopwords from the text."""
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])

def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

########## 3. Download & read data ##########
import os

projects = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']

for project in projects:

    print(f"\n=== Processing {project} ===")

    # Paths
    path = f'datasets/{project}.csv'
    datafile = f'.cleaned-text/{project}.Title+Body.csv'
    out_csv = f'.outputs/{project}_SBERT_SVM.csv'

    # ========== 3.1 Prepare cleaned data ==========
    pd_all = pd.read_csv(path)
    pd_all = pd_all.sample(frac=1, random_state=999)  # Shuffle

    # Merge Title and Body into a single column; if Body is NaN, use Title only
    pd_all['Title+Body'] = pd_all.apply(
        lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
        axis=1
    )

    # Keep only necessary columns: id, Number, sentiment, text (merged Title+Body)
    pd_tplusb = pd_all.rename(columns={
        "Unnamed: 0": "id",
        "class": "sentiment",
        "Title+Body": "text"
    })
    os.makedirs('.cleaned-text', exist_ok=True)
    pd_tplusb.to_csv(f'.cleaned-text/{project}.Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])

    # ========== 3.2 Read and clean data ==========
    data = pd.read_csv(datafile).fillna('')
    text_col = 'text'

    # Keep a copy for referencing original data if needed
    original_data = data.copy()

    # Text cleaning
    data[text_col] = data[text_col].apply(remove_html)
    data[text_col] = data[text_col].apply(remove_emoji)
    data[text_col] = data[text_col].apply(remove_stopwords)
    data[text_col] = data[text_col].apply(clean_str)

    ########## 4. Configure parameters & Start training ##########

    # ========== Hyperparameters ==========
    # Log-spaced to mirror var_smoothing in baseline
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100]
    }

    REPEATS = 10
    accuracies  = []
    precisions  = []
    recalls     = []
    f1_scores   = []
    auc_values  = []