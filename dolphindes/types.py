"""Typing for Dolphindes."""

from typing import TypeAlias, Union

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

ComplexArray: TypeAlias = NDArray[np.complexfloating]
ComplexGrid: TypeAlias = NDArray[np.complexfloating]
BoolGrid: TypeAlias = NDArray[np.bool_]
SparseDense: TypeAlias = Union[ComplexGrid, sp.sparray]