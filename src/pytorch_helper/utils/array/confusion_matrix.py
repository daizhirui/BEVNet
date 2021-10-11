import numpy as np

__all__ = [
    'multiclass_score_to_accuracy',
    'confusion_matrix',
    'confusion_matrix_to_recall',
    'confusion_matrix_to_precision',
    'confusion_matrix_to_accuracy',
    'ConfusionMatrix'
]


def multiclass_score_to_accuracy(
    score: np.ndarray, label: np.ndarray
) -> np.float:
    """ convert multiclass score accuracy.

    :param score: (N, C) array of scores
    :param label: (N, ) array of ground truth
    :return: float of accuracy
    """
    pred = np.argmax(score, axis=1)
    acc = np.sum(pred == label) / pred.size
    return acc


def confusion_matrix(
    pred: np.ndarray, gt: np.ndarray, num_classes: int
) -> np.ndarray:
    """ calculate a confusion matrix from prediction and ground truth.

    :param pred: (N, ) array of class prediction
    :param gt: (N, ) array of class ground truth
    :param num_classes: int of number of classes: C
    :return: (C, C) array of confusion matrix, where row index is the ground
        truth and column index is the prediction.
    """
    # row: ground truth
    # column: prediction
    coding = gt * num_classes + pred
    counting = np.bincount(coding, minlength=num_classes * num_classes)
    return counting.reshape(num_classes, num_classes)


def confusion_matrix_to_accuracy(conf: np.ndarray) -> np.float:
    """ calculate the accuracy from confusion matrix

    :param conf: (C, C) array of confusion matrix
    :return: float of accuracy
    """
    # accuracy: (TP + TN) / (TP + TN + FP + FN)
    return np.diag(conf).sum() / (np.sum(conf) + 1.e-16)


def confusion_matrix_to_precision(conf: np.ndarray) -> np.ndarray:
    """ calculate the precision from confusion matrix

    :param conf: (C, C) array of confusion matrix
    :return: (C, ) array of precision for each class
    """
    # precision: TP / (TP + FP)
    return np.diag(conf) / (np.sum(conf, axis=1) + 1.e-16)


def confusion_matrix_to_recall(conf: np.ndarray) -> np.ndarray:
    """ calculate the recall from confusion matrix

    :param conf: (C, C) array of confusion matrix
    :return: (C, ) array of precision for each class
    """
    # recall: TP / (TP + FN)
    return np.diag(conf) / (np.sum(conf, axis=0) + 1.e-16)


def f1(recall, precision):
    """ calculate f1 score from recall and precision

    :param recall:
    :param precision:
    :return:
    """
    return 2 * recall * precision / (recall + precision + 1.e-16)


class ConfusionMatrix:
    """ This is a helper class to calculate confusion matrix, accuracy,
    precision, recall and f1.
    """

    def __init__(self, pred, gt, num_classes):
        self.matrix = confusion_matrix(pred, gt, num_classes)

    def accuracy(self) -> float:
        return confusion_matrix_to_accuracy(self.matrix)

    def precision(self) -> np.ndarray:
        return confusion_matrix_to_precision(self.matrix)

    def recall(self) -> np.ndarray:
        return confusion_matrix_to_recall(self.matrix)

    def f1(self) -> float:
        return f1(self.recall(), self.precision())
