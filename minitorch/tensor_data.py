from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""
    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        int : Position in storage.
    """
    out = 0
    for i in range(len(index)):
        out += index[i] * strides[i]
    return out


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert an `ordinal` to an index in the `shape`. Ensures that
    enumerating position 0 ... size of a tensor produces every index exactly once.
    
    Args:
        ordinal : ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.
    """
    out_index[-1] = ordinal % shape[-1]
    shape_lst = list(shape)
    strides = strides_from_shape(shape_lst)
    for i in range(len(shape) - 1):
        out_index[i] = ordinal // strides[i]
        ordinal -= out_index[i] * strides[i]


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """
    Converts a `big_index` into a smaller `out_index` based on broadcasting rules.

    Args:
        big_index : multidimensional index of bigger tensor.
        big_shape : tensor shape of bigger tensor.
        shape : tensor shape of smaller tensor.
        out_index : multidimensional index of smaller tensor.

    Returns:
        None
    """
    len_diff = len(big_shape) - len(shape)
    for i in range(len(shape)):
        if shape[i] > 1:
            out_index[i] = big_index[len_diff + i]
        else:
            out_index[i] = 0


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape.
        shape2 : second shape.

    Returns:
        tuple : broadcasted shape.

    Raises:
        IndexingError : if the shapes cannot be broadcasted.
    """
    new_shape = [1] * max(len(shape1), len(shape2))

    for i in range(-1, -len(new_shape) - 1, -1):
        if i >= -len(shape1) and i >= -len(shape2):
            if shape1[i] != 1 and shape2[i] != 1 and shape1[i] != shape2[i]:
                raise IndexingError("Cannot broadcast shapes")
            new_shape[i] = max(shape1[i], shape2[i])
        elif i >= -len(shape1):
            new_shape[i] = shape1[i]
        else:
            new_shape[i] = shape2[i]
    return tuple(new_shape)


def strides_from_shape(shape: UserShape) -> UserStrides:
    """
    Return contiguous strides for a shape.

    Args:
        shape : shape of the tensor.

    Returns:
        tuple : strides for the shape.
    """
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    """
    A class representing a tensor's underlying data, including storage, shape, strides,
    and various utilities for tensor operations.

    Attributes:
        _storage : Storage for the tensor.
        _shape : Shape of the tensor.
        _strides : Strides for navigating the tensor.
        strides : User-friendly strides.
        shape : User-friendly shape.
        dims : Number of dimensions in the tensor.
    """

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        """
        Initialize TensorData.

        Args:
            storage : Storage for the tensor.
            shape : Shape of the tensor.
            strides : Optional strides for the tensor. If None, default strides are used.
        """
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be a tuple."
        assert isinstance(shape, tuple), "Shape must be a tuple."
        if len(strides) != len(shape):
            raise IndexingError(f"Length of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        """Convert tensor data to CUDA."""
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check if the layout is contiguous (i.e., outer dimensions have bigger strides).

        Returns:
            bool : True if the layout is contiguous.
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """
        Perform shape broadcasting between two shapes.

        Args:
            shape_a : First shape.
            shape_b : Second shape.

        Returns:
            tuple : Broadcasted shape.
        """
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """
        Convert a multidimensional index into a single-dimensional position.

        Args:
            index : Multidimensional index or integer index.

        Returns:
            int : Corresponding position in the storage.
        """
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be the size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} is not supported.")

        # Convert index to position.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """
        Generate all possible indices for the tensor based on its shape.

        Returns:
            Iterable of indices.
        """
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """
        Sample a random valid index for the tensor.

        Returns:
            tuple : Random index within tensor bounds.
        """
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """
        Get the value at the given index.

        Args:
            key : Index of the tensor.

        Returns:
            float : Value at the index.
        """
        return self._storage[self.index(key)]

    def set(self, key: UserIndex, val: float) -> None:
        """
        Set the value at the given index.

        Args:
            key : Index of the tensor.
            val : Value to set at the index.
        """
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """
        Get the underlying tensor data as a tuple.

        Returns:
            tuple : (storage, shape, strides).
        """
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            order : Permutation of the dimensions.

        Returns:
            TensorData : New tensor data with permuted dimensions.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        # Permute storage, shape, and strides.
        storage = self._storage
        shape, strides = [], []
        for i in range(len(self.shape)):
            shape.append(self.shape[order[i]])
            strides.append(self.strides[order[i]])
        return TensorData(storage, tuple(shape), tuple(strides))

    def to_string(self) -> str:
        """
        Convert tensor data to a human-readable string representation.

        Returns:
            str : String representation of the tensor.
        """
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
