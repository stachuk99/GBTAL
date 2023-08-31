import scipy
import numpy as np
import xgboost as xgb


class XGBoostWrapper:
    def __init__(self, loss_function=None, eta=0.1):
        self.loss_function = loss_function
        self.eta = eta
        self.boosting_model = None

    def fit(self, X, y, n_estimators=100, max_depth=2):
        dtrain = xgb.DMatrix(X, label=y)
        para_dict = {
            'max_depth': max_depth,
            'eta': self.eta,
            'verbosity': 1,
        }
        self.boosting_model = xgb.train(para_dict, dtrain, n_estimators, obj=self.loss_function)

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        prediction_output = self.boosting_model.predict(dtest)
        return prediction_output
