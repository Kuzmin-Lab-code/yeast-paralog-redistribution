import os
import random
from collections import OrderedDict
from typing import Dict, List

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


def average_weights(state_dicts: List[dict]) -> OrderedDict:
    """
    Averaging of input weights.

    Args:
        state_dicts: Weights to average

    Raises:
        KeyError: If states do not match

    Returns:
        Averaged weights
    """
    # source https://gist.github.com/qubvel/70c3d5e4cddcde731408f478e12ef87b
    # https://catalyst-team.github.io/catalyst/v20.11/_modules/catalyst/utils/swa.html
    params_keys = None
    for i, state_dict in enumerate(state_dicts):
        model_params_keys = list(state_dict.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(i, params_keys, model_params_keys)
            )

    average_dict = OrderedDict()
    for k in state_dicts[0].keys():
        average_dict[k] = torch.div(
            sum(state_dict[k] for state_dict in state_dicts),
            len(state_dicts),
        )

    return average_dict
