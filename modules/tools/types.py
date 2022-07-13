from pathlib import Path
from typing import *

from albumentations import BasicTransform as Transform
from albumentations import Compose
from numpy import ndarray
from omegaconf import DictConfig
from pandas.core.frame import DataFrame
from torch import Tensor

Array = Union[ndarray, Tensor]
PathT = Union[Path, str]
Transforms = Union[Compose, Transform, List[Transform]]
