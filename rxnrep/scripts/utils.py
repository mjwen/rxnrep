import copy
import random
import os
import shutil
import logging
import numpy as np
import torch
import dgl


logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience=200, silent=True):
        self.patience = patience
        self.silent = silent
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if not self.silent:
                print("EarlyStopping counter: {}/{}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def seed_torch(seed=35, cudnn_benchmark=False, cudnn_deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    dgl.random.seed(seed)


def save_checkpoints(
    state_dict_objects, misc_objects, is_best, msg=None, filename="checkpoint.pkl"
):
    """
    Save checkpoints for all objects for later recovery.

    Args:
        state_dict_objects (dict): A dictionary of objects to save. The object should
            have state_dict() (e.g. model, optimizer, ...)
        misc_objects (dict): plain python object to save
        filename (str): filename for the checkpoint

    """
    objects = copy.copy(misc_objects)
    for k, obj in state_dict_objects.items():
        objects[k] = obj.state_dict()
    torch.save(objects, filename)
    if is_best:
        shutil.copyfile(filename, "best_checkpoint.pkl")
        if msg is not None:
            logger.info(msg)


def load_checkpoints(state_dict_objects, map_location=None, filename="checkpoint.pkl"):
    """
    Load checkpoints for all objects for later recovery.

    Args:
        state_dict_objects (dict): A dictionary of objects to save. The object should
            have state_dict() (e.g. model, optimizer, ...)
    """
    checkpoints = torch.load(str(filename), map_location)
    for k, obj in state_dict_objects.items():
        state_dict = checkpoints.pop(k)
        obj.load_state_dict(state_dict)
    return checkpoints
