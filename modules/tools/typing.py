from pathlib import Path
from typing import *

from albumentations import BasicTransform as Transform
from numpy import ndarray
from pandas.core.frame import DataFrame
from torch import Tensor

Array = Union[ndarray, Tensor]
PathT = Union[Path, str]
