"""Typing for Dolphindes."""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

ComplexGrid: TypeAlias = NDArray[np.complexfloating]
BoolGrid: TypeAlias = NDArray[np.bool_]
