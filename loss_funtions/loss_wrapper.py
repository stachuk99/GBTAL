import numpy as np
import xgboost as xgb
import torch


class LossWrapper:
    def __init__(self, loss_function):
        self.loss_function = loss_function
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def loss(self, y_pred, y):
        is_xgboost = isinstance(y, xgb.DMatrix) or isinstance(y_pred, xgb.DMatrix)

        label = y.get_label() if is_xgboost else y
        label = torch.tensor(label.reshape(y_pred.shape), device=self.device, dtype=torch.float64)
        y_pred = torch.tensor(y_pred, requires_grad=True, device=self.device, dtype=torch.float64)

        grad = torch.func.grad(self.loss_function)(y_pred, label).cpu().detach().numpy()
        hess_array = []

        for i in range(y_pred.shape[0]):
            h = torch.func.hessian(self.loss_function)(y_pred[i], label[i])
            h = torch.diagonal(h)
            hess_array.append(h)
        hess = torch.stack(hess_array).cpu().detach().numpy() + np.finfo(np.float32).eps

        if is_xgboost:
            grad = grad.flatten()
            hess = hess.flatten()
        return grad, hess
