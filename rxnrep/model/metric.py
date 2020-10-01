import torch
import numpy as np
import pytorch_lightning.metrics.functional as plm
from pytorch_lightning.metrics.functional.reduction import class_reduce


class BinaryClassificationMetrics:
    def __init__(self):
        super(MultiClassificationMetrics).__init__()
        self.tps = 0
        self.fps = 0
        self.tns = 0
        self.fns = 0
        self.supps = 0

        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None

    def step(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Detach and cast to numpy arrays to avoid keeping the computational graph alive.

        See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
        """
        cm = plm.confusion_matrix(pred, target, num_classes=2)
        tn, fp, fn, tp = cm.numpy().ravel()

        self.tps += tp
        self.fps += fp
        self.tns += tn
        self.fns += fn

    def compute_metric_values(self):
        self.accuracy = (self.tps + self.tns) / (
            self.tps + self.fps + self.tns + self.fns
        )

        if self.tps + self.fps == 0:
            self.precision = 0
        else:
            self.precision = self.tps / (self.tps + self.fps)

        if self.tps + self.fns == 0:
            self.recall = 0
        else:
            self.recall = self.tps / (self.tps + self.fns)

        if self.precision + self.recall == 0:
            self.f1 = 0
        else:
            self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)

    def __str__(self):
        return "[{:.2f}|{:.2f}|{:.2f}|{:.2f}]".format(
            self.accuracy, self.precision, self.recall, self.f1
        )


class MultiClassificationMetrics:
    def __init__(self, num_classes: int):
        super(MultiClassificationMetrics).__init__()
        self.num_classes = num_classes
        self.true_positives = np.zeros(num_classes)
        self.false_positives = np.zeros(num_classes)
        self.true_negatives = np.zeros(num_classes)
        self.false_negatives = np.zeros(num_classes)
        self.supports = np.zeros(num_classes)

        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None

    def step(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Detach and cast to numpy arrays to avoid keeping the computational graph alive.
        """
        tps, fps, tns, fns, supp = plm.stat_scores_multiple_classes(
            pred, target, num_classes=self.num_classes
        )
        self.true_positives += tps.detach().numpy()
        self.false_positives += fps.detach().numpy()
        self.true_negatives += tns.detach().numpy()
        self.false_negatives += fns.detach().numpy()
        self.supports += supp.detach().numpy()

    def compute_metric_values(self, class_reduction="weighted"):
        result = accuracy_precision_recall_fbeta_from_state_scores(
            torch.as_tensor(self.true_positives),
            torch.as_tensor(self.false_positives),
            torch.as_tensor(self.true_negatives),
            torch.as_tensor(self.false_negatives),
            torch.as_tensor(self.supports),
            beta=1.0,
            class_reduction=class_reduction,
        )
        self.accuracy = result[0].item()
        self.precision = result[1].item()
        self.recall = result[2].item()
        self.f1 = result[3].item()

    def __str__(self):
        return "[{:.2f}|{:.2f}|{:.2f}|{:.2f}]".format(
            self.accuracy, self.precision, self.recall, self.f1
        )


def accuracy_precision_recall_fbeta_from_state_scores(
    tps, fps, tns, fns, sups, beta=1.0, class_reduction="micro"
):
    accuracy = class_reduce(tps, sups, sups, class_reduction=class_reduction)
    precision = class_reduce(tps, tps + fps, sups, class_reduction=class_reduction)
    recall = class_reduce(tps, tps + fns, sups, class_reduction=class_reduction)

    # We need to differentiate at which point to do class reduction
    intermidiate_reduction = "none" if class_reduction != "micro" else "micro"

    if intermidiate_reduction == "none":
        prec = class_reduce(tps, tps + fps, sups, class_reduction=intermidiate_reduction)
        rec = class_reduce(tps, tps + fns, sups, class_reduction=intermidiate_reduction)
    else:
        prec = precision
        rec = recall

    num = (1 + beta ** 2) * prec * rec
    denom = (beta ** 2) * prec + rec
    if intermidiate_reduction == "micro":
        fbeta = torch.sum(num) / torch.sum(denom)
    else:
        fbeta = class_reduce(num, denom, sups, class_reduction=class_reduction)

    return accuracy, precision, recall, fbeta
