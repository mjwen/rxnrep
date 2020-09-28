from pytorch_lightning.metrics.functional import f1_score


class F1:
    def __init__(self):
        pass

    def __call__(self, x, y):
        return f1_score(x, y)
