import numpy as np
import xgboost as xgb
from scipy.special import softmax

class Cross_Entropy:
    '''
    The class of multiclass cross entropy loss
    '''

    def __init__(self, y=None):
        '''
        :param y: placeholder for interface compatibility
        '''

    def loss(self, y_pred, y):
        # compute the prediction with sigmoid
        label = y.get_label() if isinstance(y, xgb.DMatrix) else y
        label = label.reshape(y_pred.shape)

        # sigmoid_pred = 1.0 / (1.0 + np.exp(-y_pred))
        sigmoid_pred = softmax(y_pred, axis=1)
        grad = -(label - sigmoid_pred)
        hess = sigmoid_pred * (1.0 - sigmoid_pred)
        # if isinstance(y, xgb.DMatrix):
        #     grad = grad.flatten()
        #     hess = hess.flatten()
        # else:
        #     grad = np.mean(np.sum(grad, -1))
        #     hess = np.mean(np.sum(hess, -1))
        # grad = np.mean(np.sum(-(imbalance_alpha ** label) * (label - sigmoid_pred), -1))
        # hess = np.mean(np.sum((imbalance_alpha ** label) * sigmoid_pred * (1.0 - sigmoid_pred), -1))
        return grad, hess