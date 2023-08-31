import os
import pandas as pd
import numpy as np
import sklearn
import multi_imbalance
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from multi_imbalance.utils.data import load_arff_dataset
from os import listdir
from os.path import isfile, join

dir = os.getcwd()
datasets_to_compare = [f for f in listdir(f'{dir}/data/arff/') if isfile(join(f'{dir}/data/arff/', f))]


datasets_features = {}
for dataset_name in datasets_to_compare:
    path = f'data/arff/{dataset_name}'
    X, y, cat_len = load_arff_dataset(return_non_cat_length=True, one_hot_encode=False, path=path)
    print("\n\n Dataset ***", dataset_name, "*** number of not categorical categories: ", cat_len)
    X = pd.DataFrame(X)
    enc = OneHotEncoder().fit(y.reshape(-1,1))
    y_onehot = enc.transform(y.reshape(-1, 1)).toarray()
    num_of_examples = np.sum(y_onehot)
    num_of_minority = np.min(np.sum(y_onehot, 0))
    num_of_majority = np.max(np.sum(y_onehot, 0))
    num_of_classes = len(enc.categories_[0])
    num_of_attributes = X.shape[1]
    ir = num_of_majority/num_of_minority
    datasets_features[dataset_name] = [num_of_examples, num_of_classes, num_of_minority, num_of_attributes, ir]

df = pd.DataFrame(datasets_features, index=["num_of_examples", "num_of_classes", "num_of_minority", "num_of_attributes", "ir"]).T
print(df)
df.to_csv(f"data_overview.csv")
df.to_excel(f"data_overview.xlsx")