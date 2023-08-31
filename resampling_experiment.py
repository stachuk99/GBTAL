import os
import numpy as np
import pandas as pd


from os import listdir
from os.path import isfile, join
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from multi_imbalance.utils.data import load_arff_dataset
from bgt_baseline import GBTClassifier
from xgboost_wrapper import XGBoostWrapper

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef, average_precision_score
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from multi_imbalance.resampling.global_cs import GlobalCS
from multi_imbalance.resampling.static_smote import StaticSMOTE
from multi_imbalance.resampling.soup import SOUP
from multi_imbalance.resampling.mdo import MDO

from torch.nn import CrossEntropyLoss
from loss_funtions import loss_wrapper
from datetime import datetime

import os
import multiprocessing
import sys


def fix_affinity():
    # 0 means current process
    affinity = os.sched_getaffinity(0)
    if len(affinity) != multiprocessing.cpu_count():
        print("Something has messed with CPU affinity. Current affinity is {}. Fixing".format(affinity),
              file=sys.stderr)
        os.sched_setaffinity(0, set(range(multiprocessing.cpu_count())))

        assert len(os.sched_getaffinity(0)) == multiprocessing.cpu_count(), os.sched_getaffinity(0)
    else:
        print("Affinity is OK: {}".format(affinity))

fix_affinity()


def file_size(file):
    return os.path.getsize(f'data/arff/{file}')


dir = os.getcwd()
run_type = 0

if run_type == 0:
    n_iterations = 2
    n_stratified_splits = 2
    sampling_methods = [None, GlobalCS, ClusterCentroids, SOUP, MDO]
    # sampling_methods = [StaticSMOTE,  SMOTE]# [None, GlobalCS,] # StaticSMOTE,  SMOTE, ClusterCentroids, SOUP, MDO]
    datasets_to_compare = ['hayes-roth.arff', 'thyroid-newthyroid.arff']  # , 'new_led7digit.arff', 'balance-scale.arff']
    # datasets_to_compare = [f for f in listdir(f'{dir}/data/arff/') if isfile(join(f'{dir}/data/arff/', f))]
else:
    n_iterations = 10
    n_stratified_splits = 5
    datasets_to_compare = [f for f in listdir(f'{dir}/data/arff/') if isfile(join(f'{dir}/data/arff/', f))]
    datasets_to_compare.remove('car.arff')
    sampling_methods = [None, GlobalCS, StaticSMOTE,  SMOTE, ClusterCentroids, SOUP, MDO]
    datasets_to_compare = sorted(datasets_to_compare, key=file_size)

classifiers = [GBTClassifier, XGBoostWrapper]


def compute(dataset_name, classifiers, sampling_methods, i):
    results = {}
    skf = StratifiedKFold(n_splits=n_stratified_splits, shuffle=True)
    path = f'data/arff/{dataset_name}'
    X, y, cat_len = load_arff_dataset(return_non_cat_length=True, path=path)
    enc = OneHotEncoder().fit(y.reshape(-1,1))

    for classifier in classifiers:
        for sampler in sampling_methods:
            iter_results = {}
            sampler_str = sampler.__name__ if sampler is not None else 'default'
            msg = classifier.__name__ + "\t" + sampler_str + "\t" + dataset_name.replace(
                ".arff", '') + "\t" + str(i) + "\t" + datetime.now().strftime("%m_%d-%H_%M_%S")
            print(msg)
            fold_results = []
            for train, test in skf.split(X, y):
                x_train_unsampled, x_test = X[train], X[test]
                y_train_unsampled, y_test = y[train], y[test]
                if sampler is not None:
                    s = sampler()
                    x_train, y_train = s.fit_resample(x_train_unsampled, y_train_unsampled)
                else:
                    x_train, y_train = x_train_unsampled, y_train_unsampled
                y_temp = enc.transform(y_train.reshape(-1,1)).toarray()
                y_test = enc.transform(y_test.reshape(-1,1)).toarray()
                loss = CrossEntropyLoss(reduction="sum")
                l = loss_wrapper.LossWrapper(loss)
                clf = classifier(l.loss, eta=0.1)
                clf.fit(x_train, y_temp, max_depth=2, n_estimators=100)
                y_pred = clf.predict(X[test])
                y_pred_argmax = np.argmax(y_pred, axis=1)
                fold_results.append(np.array([accuracy_score(y[test], y_pred_argmax),
                                              f1_score(y[test], y_pred_argmax, average='macro'),
                                              geometric_mean_score(y[test], y_pred_argmax, average='multiclass',
                                                                   correction=0.01),
                                              balanced_accuracy_score(y[test], y_pred_argmax),
                                              matthews_corrcoef(y[test], y_pred_argmax),
                                              average_precision_score(y_test, y_pred)]))
            mean_fold_results = np.mean(np.array(fold_results), 0)
            iter_results[f"acc_{i}"] = mean_fold_results[0]
            iter_results[f"f1_{i}"] = mean_fold_results[1]
            iter_results[f"gmean_{i}"] = mean_fold_results[2]
            iter_results[f"BA_{i}"] = mean_fold_results[3]
            iter_results[f"MCC_{i}"] = mean_fold_results[4]
            iter_results[f"AP_{i}"] = mean_fold_results[5]
            results[(classifier.__name__, sampler.__name__ if sampler is not None else 'default', dataset_name.replace(".arff", ''))] = iter_results
    return results

with Parallel(n_jobs=4) as parallel:
    execution_pairs = [(sampling_method, dataset, classifier, i) for dataset in datasets_to_compare[::-1] for sampling_method in sampling_methods for i in range(n_iterations) for classifier in classifiers]
    # print(execution_pairs)
    a = parallel(delayed(compute)(dataset_name, [classifier], [sampling_method], i) for sampling_method, dataset_name, classifier, i in execution_pairs)

result_dict = a[0]
for r in a[1:]:
    updated = False
    keys = result_dict.keys()
    for k in keys:
        if k in r:
            result_dict[k].update(r[k])
            updated = True
            break
    if not updated:
        result_dict.update(r)
df_results = pd.DataFrame(result_dict).T

df_results['avg_acc'] = df_results.filter(regex='acc', axis=1).mean(1)
df_results['avg_f1'] = df_results.filter(regex='f1', axis=1).mean(1)
df_results['avg_gmean'] = df_results.filter(regex='gmean', axis=1).mean(1)
df_results['avg_BA'] = df_results.filter(regex='BA', axis=1).mean(1)
df_results['avg_MCC'] = df_results.filter(regex='MCC', axis=1).mean(1)
df_results['avg_AP'] = df_results.filter(regex='AP', axis=1).mean(1)
print(df_results)

current_dateTime = datetime.now()
date_str = current_dateTime.strftime("%m_%d-%H_%M_%S")
df_results.to_csv(f"results_sampling{date_str}.csv")
df_results.to_excel(f"results_sampling{date_str}.xlsx")
