"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """Stores the history of `Function` operations that were used to construct the current Variable."""

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a multidimensional array that supports autodifferentiation."""

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        """Initialize a Tensor.

        Args:
        ----
            v: TensorData representing the underlying data.
            back: Optional history for backpropagation.
            name: Optional name for the tensor.
            backend: Backend used for tensor operations (e.g., NumPy, CUDA).

        """
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        self.name = name if name is not None else str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Sets the gradient requirement for this tensor."""
        self.history = History()

    def requires_grad(self) -> bool:
        """Checks if this tensor requires gradients."""
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Converts the tensor to a NumPy array.

        Returns
        -------
            A NumPy array representing the tensor data.

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    # Properties
    @property
    def shape(self) -> UserShape:
        """Returns the shape of the tensor."""
        return self._tensor.shape

    @property
    def size(self) -> int:
        """Returns the size of the tensor (total number of elements)."""
        return self._tensor.size

    @property
    def dims(self) -> int:
        """Returns the number of dimensions of the tensor."""
        return self._tensor.dims

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Converts a Python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            return Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            return b

    # Functions
    def __add__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, -self._ensure_tensor(b))

    def __mul__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, self._ensure_tensor(b))

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3."""
        return MatMul.apply(self, b)

    def __lt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b: TensorLike) -> Tensor:  # type: ignore[override]
        return EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self._ensure_tensor(b), self)

    def __neg__(self) -> Tensor:
        return Neg.apply(self)

    def __radd__(self, b: TensorLike) -> Tensor:
        return self + b

    def __rmul__(self, b: TensorLike) -> Tensor:
        return self * b

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Checks if all elements along a dimension are true."""
        if dim is None:
            return All.apply(self.view(self.size), self._ensure_tensor(0))
        return All.apply(self, self._ensure_tensor(dim))

    def is_close(self, y: Tensor) -> Tensor:
        """Checks if two tensors are element-wise close."""
        return IsClose.apply(self, y)

    def sigmoid(self) -> Tensor:
        """Applies the sigmoid function element-wise."""
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Applies the ReLU function element-wise."""
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Applies the natural logarithm element-wise."""
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Applies the exponential function element-wise."""
        return Exp.apply(self)

    def item(self) -> float:
        """Converts a 1-element tensor to a Python float."""
        assert self.size == 1
        return self[0]

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Computes the sum over a dimension.

        Args:
        ----
            dim: Dimension along which to sum. If None, sums over all elements.

        Returns:
        -------
            The sum of elements along the given dimension or the total sum.

        """
        if dim is None:
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Computes the mean over a dimension.

        Args:
        ----
            dim: Dimension along which to compute the mean. If None, computes the mean over all elements.

        Returns:
        -------
            The mean of elements along the given dimension or the total mean.

        """
        if dim is not None:
            return self.sum(dim) / self.shape[dim]
        return self.sum() / self.size

    def permute(self, *order: int) -> Tensor:
        """Permutes the dimensions of the tensor according to the given order."""
        return Permute.apply(self, tensor(list(order)))

    def view(self, *shape: int) -> Tensor:
        """Reshapes the tensor.

        Args:
        ----
            shape: New shape of the tensor.

        Returns:
        -------
            A new tensor with the given shape.

        """
        return View.apply(self, tensor(list(shape)))

    def contiguous(self) -> Tensor:
        """Returns a contiguous tensor with the same data."""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Creates a new tensor from the provided data."""
        # Fix to ensure compatibility with Storage
        if isinstance(storage, list):
            storage = [float(x) for x in storage]
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Expands a tensor to allow for backpropagation over broadcasting."""
        if self.shape == other.shape:
            return other

        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)

        if self.shape == true_shape:
            return buf

        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Creates a tensor filled with zeros.

        Args:
        ----
            shape: The shape of the tensor. If None, uses the shape of the current tensor.

        Returns:
        -------
            A new tensor filled with zeros.

        """

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Returns the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detaches the tensor from its computation history."""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backpropagation
    def accumulate_derivative(self, x: Any) -> None:
        """Adds the given value to the accumulated derivative of this variable."""
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0] * int(operators.prod(self.shape)), self.shape, backend=self.backend
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """Checks if this variable was created by the user (no `last_fn`)."""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Checks if this tensor is a constant (no history)."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the parent variables used to compute this tensor."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes the chain rule to propagate gradients backward."""
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Performs backpropagation to compute gradients."""
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def zero_grad_(self) -> None:
        """Resets the gradient of this tensor."""
        self.grad = None
