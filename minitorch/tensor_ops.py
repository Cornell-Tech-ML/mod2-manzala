from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Optional, Type
import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    """Protocol for mapping a function over a tensor."""

    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Applies a mapping function to the input tensor.

        Args:
        ----
            x: The input tensor to apply the function on.
            out: Optional output tensor to store the result. If not provided, a new tensor is created.

        Returns:
        -------
            A tensor with the mapping function applied to the input tensor.

        """
        ...


class TensorOps:
    """Class for implementing high-level tensor operations."""

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order function to map a function over a tensor.

        Args:
        ----
            fn: A function that takes a float and returns a float.

        Returns:
        -------
            A callable that maps the function over the input tensor and returns a new tensor.

        """

        def operation(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = x.zeros(x.shape)
            # Apply the function here (this should be actual logic)
            return out

        return operation

    @staticmethod
    def cmap(fn: Callable[[float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Higher-order function to map a function over two tensors element-wise.

        Args:
        ----
            fn: A function that takes two floats and returns a float.

        Returns:
        -------
            A callable that applies the function over the two input tensors and returns a new tensor.

        """

        def operation(x: Tensor, y: Tensor) -> Tensor:
            out = x.zeros(shape_broadcast(x.shape, y.shape))
            # Apply the function here (this should be actual logic)
            return out

        return operation

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Higher-order function to zip two tensors element-wise using a function.

        Args:
        ----
            fn: A function that takes two floats and returns a float.

        Returns:
        -------
            A callable that applies the function to the elements of two input tensors and returns a new tensor.

        """

        def operation(a: Tensor, b: Tensor) -> Tensor:
            out = a.zeros(shape_broadcast(a.shape, b.shape))
            # Apply the function here (this should be actual logic)
            return out

        return operation

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Higher-order function to reduce a tensor along a specified dimension.

        Args:
        ----
            fn: A function that takes two floats and returns a float.
            start: Starting value for the reduction.

        Returns:
        -------
            A callable that reduces the input tensor along a given dimension.

        """

        def operation(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start
            # Apply the reduction logic here
            return out

        return operation

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Performs matrix multiplication on two tensors.

        Args:
        ----
            a: The first input tensor.
            b: The second input tensor.

        Returns:
        -------
            A tensor resulting from the matrix multiplication.

        Raises:
        ------
            NotImplementedError: If matrix multiplication is not implemented.

        """
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    """Class representing a tensor backend using high-level tensor operations."""

    def __init__(self, ops: Type[TensorOps]):
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
        ----
            ops: A class implementing tensor operations like map, zip, and reduce.

        """
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.id_cmap = ops.cmap(operators.id)
        self.inv_map = ops.map(operators.inv)

        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    """Simple implementation of tensor operations."""

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map function.

        Args:
        ----
            fn: A function that takes a float and returns a float.

        Returns:
        -------
            A callable that maps the function over the input tensor and returns a new tensor.

        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Higher-order tensor zip function.

        Args:
        ----
            fn: A function that takes two floats and returns a float.

        Returns:
        -------
            A callable that zips two tensors using the function and returns a new tensor.

        """
        f = tensor_zip(fn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Higher-order tensor reduce function.

        Args:
        ----
            fn: A function that takes two floats and returns a float.
            start: Starting value for the reduction.

        Returns:
        -------
            A callable that reduces the input tensor along a specified dimension.

        """
        f = tensor_reduce(fn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start
            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Performs matrix multiplication on two tensors.

        Args:
        ----
            a: The first input tensor.
            b: The second input tensor.

        Returns:
        -------
            A tensor resulting from the matrix multiplication.

        Raises:
        ------
            NotImplementedError: If matrix multiplication is not implemented.

        """
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


def tensor_map(fn: Callable[[float], float]) -> Any:
    """Low-level implementation of tensor map.

    Args:
    ----
        fn: A function that takes a float and returns a float.

    Returns:
    -------
        A function to map a tensor element-wise.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_idx = np.array(out_shape)
        in_idx = np.array(in_shape)

        for i in range(len(out)):
            to_index(i, out_shape, out_idx)
            broadcast_index(out_idx, out_shape, in_shape, in_idx)
            out[i] = fn(in_storage[index_to_position(in_idx, in_strides)])

    return _map


def tensor_zip(fn: Callable[[float, float], float]) -> Any:
    """Low-level implementation of tensor zip.

    Args:
    ----
        fn: A function that takes two floats and returns a float.

    Returns:
    -------
        A function to zip two tensors element-wise.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_idx = np.array(out_shape)
        a_idx = np.array(a_shape)
        b_idx = np.array(b_shape)

        for i in range(len(out)):
            to_index(i, out_shape, out_idx)
            broadcast_index(out_idx, out_shape, a_shape, a_idx)
            broadcast_index(out_idx, out_shape, b_shape, b_idx)
            out[i] = fn(
                a_storage[index_to_position(a_idx, a_strides)],
                b_storage[index_to_position(b_idx, b_strides)],
            )

    return _zip


def tensor_reduce(fn: Callable[[float, float], float]) -> Any:
    """Low-level implementation of tensor reduce.

    Args:
    ----
        fn: A function that takes two floats and returns a float.

    Returns:
    -------
        A function to reduce a tensor along a specified dimension.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        out_idx = np.array(out_shape)

        for i in range(len(out)):
            to_index(i, out_shape, out_idx)
            for j in range(a_shape[reduce_dim]):
                a_idx = np.array(out_idx)
                a_idx[reduce_dim] = j
                out[i] = fn(out[i], a_storage[index_to_position(a_idx, a_strides)])

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
