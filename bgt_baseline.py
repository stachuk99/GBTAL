import scipy
import numpy as np
from scipy.special import softmax



class Node(object):
    def __init__(self, depth=1, calculate_leaf_value=None):
        self.left = None
        self.right = None
        self.boundary = [None, None]
        self.value = None
        self.depth = depth
        if calculate_leaf_value is None:
            self.calculate_leaf_value = self.mean_val_leaf
        else:
            self.calculate_leaf_value = calculate_leaf_value

    def mean_val_leaf(self, X, y, last_predicted):
        return np.mean(y)

    def fit(self, X, y, last_predicted, max_depth=None):
        if max_depth is not None and self.depth > max_depth:
            self.value = self.calculate_leaf_value(X, y, last_predicted)
            return

        if len(y) <= 1:
            best_loss = 0
        else:
            best_loss = np.var(y)
        for i in range(np.shape(X)[1]):
            candidates = np.unique(X[:, i])
            for candidate in candidates:
                left_y, right_y = y[X[:, i] < candidate], y[X[:, i] >= candidate]
                if len(left_y) <= 1:
                    left_var = 0
                else:
                    left_var = np.var(left_y)

                if len(right_y) <= 1:
                    right_var = 0
                else:
                    right_var = np.var(right_y)

                candidate_loss = left_var * len(left_y) + right_var * len(right_y)
                candidate_loss /= len(y)
                if candidate_loss < best_loss:
                    self.boundary = [i, candidate]
                    best_loss = candidate_loss

        if self.boundary[0] is not None:
            i, split_val = self.boundary
            left_x = X[X[:, i] < split_val]
            right_x = X[X[:, i] >= split_val]
        if self.boundary[0] is not None and left_x.size != 0 and right_x.size != 0:
            self.left = Node(self.depth + 1, self.calculate_leaf_value)
            self.right = Node(self.depth + 1, self.calculate_leaf_value)
            self.left.fit(left_x, y[X[:, i] < split_val], last_predicted[X[:, i] < split_val],
                          max_depth)
            self.right.fit(right_x, y[X[:, i] >= split_val], last_predicted[X[:, i] >= split_val],
                           max_depth)
        else:
            self.value = self.calculate_leaf_value(X, y, last_predicted)

    def predict_an_instance(self, x):
        if self.value is not None:
            return self.value
        else:
            if x[self.boundary[0]] < self.boundary[1]:
                return self.left.predict(x)
            else:
                return self.right.predict(x)

    def predict(self, X):
        if X.ndim == 1:
            return self.predict_an_instance(X)
        y = [self.predict_an_instance(x).flatten() for x in X]
        return np.array(y)

    def get_leaf_values(self):
        if self.value is not None:
            return self.value
        else:
            return [self.left.get_leaf_values(), self.right.get_leaf_values()]


class GBTClassifier(object):
    def __init__(self, loss_function=None, eta=0.1):
        self.eta = eta
        self.trees = []
        self.initial_model = None
        self.loss_function = loss_function

    def calc_leaf(self, X, y, logits):
        grad, hess = self.loss_function(y, logits)
        leaf = grad / hess
        leaf = np.mean(np.clip(leaf, -10, 10), 0)
        return leaf

    def fit(self, X, y, n_estimators=100, max_depth=2):
        self.initial_model = np.log(np.mean(y, axis=0) / (1 - np.mean(y, axis=0)))
        logits = np.full(y.shape, self.initial_model)
        for i in range(n_estimators):
            loss = self.loss_function(logits, y)
            r = -loss[0]
            r = np.clip(r, -10, 10)
            node = Node(calculate_leaf_value=self.calc_leaf)
            node.fit(X, r, logits, max_depth=max_depth)
            self.trees.append(node)
            logits += node.predict(X) * self.eta


    def predict(self, X):
        y_pred = np.array([node.predict(X) for node in self.trees]).sum(axis=0)
        return softmax(self.initial_model + y_pred * self.eta, 1)
