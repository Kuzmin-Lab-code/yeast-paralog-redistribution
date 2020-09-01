import numpy as np

Array = np.ndarray


def accuracy(y_true: Array, y_pred: Array, top: int = 1):
    """
    Top-K accuracy
    :param y_true: array of true labels
    :param y_pred: array of predicted labels
    :param top: top predictions in which to look
    :return: top-k accuracy
    """
    return np.mean([y in pred.argsort()[::-1][:top] for y, pred in zip(y_true, y_pred)])
