import numpy as np


def MSE(yhat, y):
    yhat = yhat.reshape(-1)
    y = y.reshape(-1)
    return np.mean((yhat - y) ** 2)


def RMSE(yhat, y):
    return np.sqrt(MSE(yhat, y))


def MAPE(yhat, y):
    yhat = yhat.reshape(-1)
    y = y.reshape(-1)
    return np.mean(np.abs(yhat - y) / (np.abs(y) + 1e-8))

def MAE(yhat, y):
    yhat = yhat.reshape(-1)
    y = y.reshape(-1)
    return np.mean(np.abs(yhat - y))


def RRSE(yhat, y):
    yhat = yhat.reshape(-1)
    y = y.reshape(-1)
    y_mean = np.mean(y)
    return np.sqrt(np.sum((yhat - y)**2) / np.sum((y - y_mean)**2))
