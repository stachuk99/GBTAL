import os
import sys
import numpy as np
import pandas as pd

import torch


from os import listdir
from os.path import isfile, join
from datetime import datetime
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from multi_imbalance.utils.data import load_arff_dataset
from bgt_baseline import GBTClassifier
from xgboost_wrapper import XGBoostWrapper

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef, average_precision_score
from imblearn.metrics import geometric_mean_score
from loss_funtions import loss_wrapper, dice_loss, focal_loss_kornia, LDAMLoss
from torch.nn import CrossEntropyLoss
import multiprocessing
import sys
dir = os.getcwd()


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
    loss_to_compare = [
        (None, {}),
        (CrossEntropyLoss, {"reduction": "sum", "weight": 0}),
        (focal_loss_kornia.FocalLossKornia, {"alpha": 0.25, "gamma": 2, "reduction": "sum"}),
        (dice_loss.DiceLoss,
         {"alpha": 0.25, "square_denominator": True, "reduction": "sum", "index_label_position": False,
          "smooth": 1e-3}),
        (LDAMLoss.LDAMLoss, {"cls_num_list": 0, "max_m": 0.5, "s": 15}),
    ]
    datasets_to_compare = ['hayes-roth.arff', 'thyroid-newthyroid.arff']  # , 'new_led7digit.arff', 'balance-scale.arff']
    # datasets_to_compare = [f for f in listdir(f'{dir}/data/arff/') if isfile(join(f'{dir}/data/arff/', f))]

else:
    n_iterations = 10
    n_stratified_splits = 5
    datasets_to_compare = [f for f in listdir(f'{dir}/data/arff/') if isfile(join(f'{dir}/data/arff/', f))]
    datasets_to_compare = sorted(datasets_to_compare, key=file_size)
    loss_to_compare = [
        (None, {}),
        (CrossEntropyLoss, {"reduction": "sum", "weight": 0}),
        (CrossEntropyLoss, {"reduction": "sum"}),
        (focal_loss_kornia.FocalLossKornia, {"alpha": 0.25, "gamma": 2, "reduction": "sum"}),
        (focal_loss_kornia.FocalLossKornia, {"alpha": 0.25, "gamma": 1, "reduction": "sum"}),
        (focal_loss_kornia.FocalLossKornia, {"alpha": 0.25, "gamma": 3, "reduction": "sum"}),
        (focal_loss_kornia.FocalLossKornia, {"alpha": 0.5, "gamma": 2, "reduction": "sum"}),
        (focal_loss_kornia.FocalLossKornia, {"alpha": 0.5, "gamma": 1, "reduction": "sum"}),
        (focal_loss_kornia.FocalLossKornia, {"alpha": 0.5, "gamma": 3, "reduction": "sum"})
        # (dice_loss.DiceLoss,
        #  {"alpha": 0.25, "square_denominator": True, "reduction": "sum", "index_label_position": False,
        #   "smooth": 1e-3}),
        # (dice_loss.DiceLoss,
        #  {"alpha": 0.25, "square_denominator": True, "reduction": "sum", "index_label_position": False,
        #   "smooth": 1e-4}),
        # (dice_loss.DiceLoss,
        #  {"alpha": 0.25, "square_denominator": True, "reduction": "sum", "index_label_position": False,
        #   "smooth": 1e-5}),
        # (dice_loss.DiceLoss,
        #  {"alpha": 0.5, "square_denominator": True, "reduction": "sum", "index_label_position": False, "smooth": 1e-3}),
        # (dice_loss.DiceLoss,
        #  {"alpha": 0.5, "square_denominator": True, "reduction": "sum", "index_label_position": False, "smooth": 1e-4}),
        # (dice_loss.DiceLoss,
        #  {"alpha": 0.5, "square_denominator": True, "reduction": "sum", "index_label_position": False, "smooth": 1e-5})
        # (LDAMLoss.LDAMLoss, {"cls_num_list": 0, "max_m": 0.5, "s": 15}),
        # (LDAMLoss.LDAMLoss, {"cls_num_list": 0, "max_m": 0.5, "s": 30}),
        # (LDAMLoss.LDAMLoss, {"cls_num_list": 0, "max_m": 0.5, "s": 45}),
        # (LDAMLoss.LDAMLoss, {"cls_num_list": 0, "max_m": 0.25, "s": 15}),
        # (LDAMLoss.LDAMLoss, {"cls_num_list": 0, "max_m": 0.25, "s": 30}),
        # (LDAMLoss.LDAMLoss, {"cls_num_list": 0, "max_m": 0.25, "s": 45}),
    ]


classifiers = [GBTClassifier, XGBoostWrapper]


def compute(dataset_name, classifiers, loss_to_compare, i):
    fix_affinity()
    results = {}
    skf = StratifiedKFold(n_splits=n_stratified_splits, shuffle=True)
    path = f'data/arff/{dataset_name}'
    X, y, cat_len = load_arff_dataset(return_non_cat_length=True, path=path)
    enc = OneHotEncoder().fit(y.reshape(-1,1))

    for classifier in classifiers:
        for loss, loss_param in loss_to_compare:
            iter_results = {}
            msg = classifier.__name__ + "\t" + str(loss) + "\t" + str(loss_param)+ "\t" + dataset_name.replace(
                ".arff", '') + "\t" + str(i) + "\t" + datetime.now().strftime("%m_%d-%H_%M_%S")
            print(msg)
            fold_results = []
            for train, test in skf.split(X, y):
                y_temp = enc.transform(y.reshape(-1,1)).toarray()
                if 'cls_num_list' in loss_param:
                    loss_param["cls_num_list"] = np.sum(y_temp, 0)
                if "weight" in loss_param:
                    loss_param["weight"] = torch.tensor(np.sum(y_temp) /np.sum(y_temp, 0))
                if loss is not None:
                    lf = loss(**loss_param)
                    l = loss_wrapper.LossWrapper(lf).loss
                else:
                    if classifier.__name__ == "GBTClassifier":
                        fold_results = np.zeros((n_stratified_splits, 6))
                        break
                    l = None
                clf = classifier(loss_function=l, eta=0.1)
                clf.fit(X[train], y_temp[train], max_depth=2, n_estimators=100)
                y_pred = clf.predict(X[test])
                y_pred_argmax = np.argmax(y_pred, axis=1)
                fold_results.append(np.array([accuracy_score(y[test], y_pred_argmax),
                        f1_score(y[test], y_pred_argmax, average='macro'),
                        geometric_mean_score(y[test], y_pred_argmax, average='multiclass', correction=0.01),
                        balanced_accuracy_score(y[test], y_pred_argmax),
                        matthews_corrcoef(y[test], y_pred_argmax),
                        average_precision_score(y_temp[test], y_pred)]))
            mean_fold_results = np.mean(np.array(fold_results), 0)
            iter_results[f"acc_{i}"] = mean_fold_results[0]
            iter_results[f"f1_{i}"] = mean_fold_results[1]
            iter_results[f"gmean_{i}"] = mean_fold_results[2]
            iter_results[f"BA_{i}"] = mean_fold_results[3]
            iter_results[f"MCC_{i}"] = mean_fold_results[4]
            iter_results[f"AP_{i}"] = mean_fold_results[5]
            results[(classifier.__name__, loss.__name__ if loss is not None else 'default', str(loss_param), dataset_name.replace(".arff", ''))] = iter_results
    return results


with Parallel(n_jobs=8) as parallel:
    execution_pairs = [(loss, dataset, classifier, i) for dataset in datasets_to_compare[::-1] for loss in loss_to_compare for i in range(n_iterations) for classifier in classifiers]
    print(execution_pairs)
    a = parallel(delayed(compute)(dataset_name, [classifier], [loss],i) for loss, dataset_name, classifier, i in execution_pairs )

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
date_str= current_dateTime.strftime("%m_%d-%H_%M_%S")
df_results.to_csv(f"results_torch{date_str}.csv")
df_results.to_excel(f"results_torch{date_str}.xlsx")