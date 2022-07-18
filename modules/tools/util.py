import glob
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
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


def load_cfg_and_checkpoint(run_path: str) -> Tuple[DictConfig, Dict]:
    run_path = Path(run_path)
    if not run_path.exists():
        raise ValueError(f"{run_path} does not exist")
    if not run_path.is_dir():
        raise ValueError(
            f"{run_path} is not a directory. Make sure you provide a path to a directory created by Hydra"
        )

    checkpoints = glob.glob(str(run_path / "**/**/checkpoints/*.ckpt"))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {run_path}")

    weights = [torch.load(c)["state_dict"] for c in checkpoints]
    weights = average_weights(weights)

    cfg = OmegaConf.load(run_path / ".hydra" / "config.yaml")

    return cfg, weights


def load_checkpoint_from_run_path(
    model: torch.nn.Module, path: Union[str, Path], select_checkpoint: str = "average", strict: bool = True,
) -> torch.nn.Module:
    path = Path(path)
    # Load checkpoints
    if select_checkpoint == "average":
        checkpoints = glob.glob(str(path / "**/**/checkpoints/*.ckpt"))
    elif select_checkpoint == "last":
        checkpoints = glob.glob(str(path / "**/**/checkpoints/last.ckpt"))
    elif select_checkpoint == "best":
        # todo not guaranteed to be best! double-check top-k saving pattern
        checkpoints = [glob.glob(str(path / "**/**/checkpoints/*.ckpt"))[-2]]
    else:
        raise ValueError(
            f"Checkpoint {select_checkpoint} is not supported, choose from [average, last, best]"
        )

    checkpoints = [torch.load(c)["state_dict"] for c in checkpoints]
    checkpoints = average_weights(checkpoints)
    model.load_state_dict(checkpoints, strict=strict)
    return model
