import pandas as pd
import numpy as np
import ast
from scipy.stats import wilcoxon

projects = ['tensorflow', 'pytorch', 'keras', 'incubator-mxnet', 'caffe']
labels   = ['TensorFlow', 'PyTorch', 'Keras', 'MXNet', 'Caffe']

print("------------------------------------------")
print(f"| {'Project'} | {'NB F1'} | {'SVM F1'} | {'p-value'} | {'Sig.'} |")
print("------------------------------------------")

for project, label in zip(projects, labels):
    nb  = pd.read_csv(f'.outputs/{project}_TFIDF_NB.csv')
    svm = pd.read_csv(f'.outputs/{project}_SBERT_SVM.csv')

    nb_f1s  = ast.literal_eval(nb.iloc[0]['CV_list(F1)'])
    svm_f1s = ast.literal_eval(svm.iloc[0]['CV_list(F1)'])

    nb_mean  = np.mean(nb_f1s)
    svm_mean = np.mean(svm_f1s)

    stat, p = wilcoxon(nb_f1s, svm_f1s, alternative='less')  # Test if NB F1 < SVM F1
    sig = 'Yes' if p < 0.05 else 'No'

    print(f"| {label} | {nb_mean:.4f} | {svm_mean:.4f} | {p:.4f} | {sig} |")
print("------------------------------------------")
