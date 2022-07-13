import os
import random
from typing import Dict

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import SubsetRandomSampler


class SubsetSampler(SubsetRandomSampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))


def fix_seed(seed: int = 65) -> None:
    """
    Fix all random seeds for reproducibility
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_cfg(cfg: DictConfig) -> Dict:
    """
    Flatten nested config dict
    :param cfg: omagaconf config
    :return: flattened dict config
    """
    d_cfg = {}
    for group, group_dict in dict(cfg).items():
        if isinstance(group_dict, DictConfig):
            for param, value in dict(group_dict).items():
                d_cfg[f"{group}.{param}"] = value
        else:
            d_cfg[group] = group_dict
    return d_cfg
