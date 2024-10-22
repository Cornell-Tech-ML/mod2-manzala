"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, List, Tuple

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> Tuple[Any, ...]:
    """Turn a possible value into a tuple."""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    """Base class for all differentiable functions.
    Provides the apply method for automatic differentiation.
    """

    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        """Execute the backward pass to compute gradients.

        Args:
        ----
            ctx: Context object storing intermediate values from forward pass.
            grad_out: Tensor representing the gradient of the output.

        Returns:
        -------
            Tuple of Tensors containing the gradients for each input.

        """
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        """Execute the forward pass.

        Args:
        ----
            ctx: Context object to store intermediate values for backward pass.
            *inps: Input tensors.

        Returns:
        -------
            Resultant Tensor.

        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Apply the function to the input tensors.

        Args:
        ----
            *vals: Input tensors.

        Returns:
        -------
            Resultant Tensor.

        """
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        ctx = Context(not need_grad)
        c = cls._forward(ctx, *raw_vals)

        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    """Negation function for tensors."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for negation.

        Args:
        ----
            ctx: Context to store intermediate values.
            t1: Input tensor.

        Returns:
        -------
            Negated tensor.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for negation.

        Args:
        ----
            ctx: Context storing intermediate values.
            grad_output: Gradient of the output.

        Returns:
        -------
            Gradient of the input.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    """Inversion function for tensors."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for inversion.

        Args:
        ----
            ctx: Context to store intermediate values.
            t1: Input tensor.

        Returns:
        -------
            Inverted tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for inversion.

        Args:
        ----
            ctx: Context storing intermediate values.
            grad_output: Gradient of the output.

        Returns:
        -------
            Gradient of the input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    """Addition function for tensors."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for addition.

        Args:
        ----
            ctx: Context to store intermediate values.
            t1: First input tensor.
            t2: Second input tensor.

        Returns:
        -------
            Sum of the two tensors.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for addition.

        Args:
        ----
            ctx: Context storing intermediate values.
            grad_output: Gradient of the output.

        Returns:
        -------
            Gradients for each input.

        """
        return grad_output, grad_output


class Mul(Function):
    """Multiplication function for tensors."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Forward pass for multiplication.

        Args:
        ----
            ctx: Context to store intermediate values.
            a: First input tensor.
            b: Second input tensor.

        Returns:
        -------
            Product of the two tensors.

        """
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for multiplication.

        Args:
        ----
            ctx: Context storing intermediate values.
            grad_output: Gradient of the output.

        Returns:
        -------
            Gradients for each input.

        """
        a, b = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, b), grad_output.f.mul_zip(
            grad_output, a
        )


class Sigmoid(Function):
    """Sigmoid activation function for tensors."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the sigmoid function.

        Args:
        ----
            ctx: Context to store intermediate values.
            t1: Input tensor.

        Returns:
        -------
            Sigmoid of the input tensor.

        """
        output = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the sigmoid function.

        Args:
        ----
            ctx: Context storing intermediate values.
            grad_output: Gradient of the output.

        Returns:
        -------
            Gradient of the input.

        """
        (output,) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, output * (-output + 1.0))


class ReLU(Function):
    """ReLU activation function for tensors."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for ReLU.

        Args:
        ----
            ctx: Context to store intermediate values.
            t1: Input tensor.

        Returns:
        -------
            ReLU of the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for ReLU.

        Args:
        ----
            ctx: Context storing intermediate values.
            grad_output: Gradient of the output.

        Returns:
        -------
            Gradient of the input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):
    """Logarithm function for tensors."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for logarithm.

        Args:
        ----
            ctx: Context to store intermediate values.
            t1: Input tensor.

        Returns:
        -------
            Logarithm of the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for logarithm.

        Args:
        ----
            ctx: Context storing intermediate values.
            grad_output: Gradient of the output.

        Returns:
        -------
            Gradient of the input.

        """
        (t1,) = ctx.saved_tensors
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    """Exponential function for tensors."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the exponential function.

        Args:
        ----
            ctx: Context to store intermediate values.
            t1: Input tensor.

        Returns:
        -------
            Exponential of the input tensor.

        """
        output = t1.f.exp_map(t1)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the exponential function.

        Args:
        ----
            ctx: Context storing intermediate values.
            grad_output: Gradient of the output.

        Returns:
        -------
            Gradient of the input.

        """
        (output,) = ctx.saved_tensors
        return grad_output.f.mul_zip(grad_output, output)


class Sum(Function):
    """Summation function for tensors."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for summation.

        Args:
        ----
            ctx: Context to store intermediate values.
            a: Input tensor.
            dim: Dimension to sum over.

        Returns:
        -------
            Summed tensor along the given dimension.

        """
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for summation.

        Args:
        ----
            ctx: Context storing intermediate values.
            grad_output: Gradient of the output.

        Returns:
        -------
            Gradient of the input.

        """
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class All(Function):
    """Multiplication over a tensor (all elements must be True)."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for `all` operation.

        Args:
        ----
            ctx: Context to store intermediate values.
            a: Input tensor.
            dim: Dimension to operate over.

        Returns:
        -------
            Tensor with `all` operation applied.

        """
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class LT(Function):
    """Less than operation for tensors."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Forward pass for less than comparison.

        Args:
        ----
            ctx: Context to store intermediate values.
            a: First input tensor.
            b: Second input tensor.

        Returns:
        -------
            Tensor of boolean values where a < b.

        """
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for less than comparison.

        Args:
        ----
            ctx: Context storing intermediate values.
            grad_output: Gradient of the output.

        Returns:
        -------
            Zeros for both inputs as the gradient of a comparison.

        """
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class EQ(Function):
    """Equality operation for tensors."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Forward pass for equality comparison.

        Args:
        ----
            ctx: Context to store intermediate values.
            a: First input tensor.
            b: Second input tensor.

        Returns:
        -------
            Tensor of boolean values where a == b.

        """
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for equality comparison.

        Args:
        ----
            ctx: Context storing intermediate values.
            grad_output: Gradient of the output.

        Returns:
        -------
            Zeros for both inputs as the gradient of a comparison.

        """
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class IsClose(Function):
    """Approximate equality operation for tensors."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Forward pass for approximate equality.

        Args:
        ----
            ctx: Context to store intermediate values.
            a: First input tensor.
            b: Second input tensor.

        Returns:
        -------
            Tensor of boolean values where abs(a - b) < tolerance.

        """
        return a.f.is_close_zip(a, b)


class Permute(Function):
    """Permutes the dimensions of a tensor."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Forward pass for permutation of tensor dimensions.

        Args:
        ----
            ctx: Context to store intermediate values.
            a: Input tensor.
            order: Permutation order.

        Returns:
        -------
            Permuted tensor.

        """
        new_order = tuple(order.to_numpy().astype(int))
        ctx.save_for_backward(new_order)
        new_a = a._tensor.permute(*new_order)
        return a._new(new_a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for permutation of tensor dimensions.

        Args:
        ----
            ctx: Context storing intermediate values.
            grad_output: Gradient of the output.

        Returns:
        -------
            Permuted gradient tensor.

        """
        (original_order,) = ctx.saved_values
        order = np.argsort(original_order)
        new_a = grad_output._tensor.permute(*order)
        return minitorch.Tensor.make(
            new_a._storage, new_a.shape, backend=grad_output.backend
        ), 0.0


class View(Function):
    """Changes the shape of a tensor."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Forward pass for reshaping a tensor.

        Args:
        ----
            ctx: Context to store intermediate values.
            a: Input tensor.
            shape: New shape.

        Returns:
        -------
            Reshaped tensor.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for reshaping a tensor.

        Args:
        ----
            ctx: Context storing intermediate values.
            grad_output: Gradient of the output.

        Returns:
        -------
            Reshaped gradient tensor.

        """
        (original,) = ctx.saved_values
        return minitorch.Tensor.make(
            grad_output._tensor._storage, original, backend=grad_output.backend
        ), 0.0


class Copy(Function):
    """Creates a copy of a tensor."""

    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Forward pass to create a copy.

        Args:
        ----
            ctx: Context to store intermediate values.
            a: Input tensor.

        Returns:
        -------
            Copied tensor.

        """
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the copy operation.

        Args:
        ----
            ctx: Context storing intermediate values.
            grad_output: Gradient of the output.

        Returns:
        -------
            Gradient of the input.

        """
        return grad_output


class MatMul(Function):
    """Matrix multiplication for tensors."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for matrix multiplication.

        Args:
        ----
            ctx: Context to store intermediate values.
            t1: First input tensor.
            t2: Second input tensor.

        Returns:
        -------
            Matrix product of the two tensors.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for matrix multiplication.

        Args:
        ----
            ctx: Context storing intermediate values.
            grad_output: Gradient of the output.

        Returns:
        -------
            Gradients of the input tensors.

        """
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape: Shape of tensor.
        backend: Tensor backend.

    Returns:
    -------
        New tensor.

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape: Shape of tensor.
        backend: Tensor backend.
        requires_grad: Whether the tensor requires gradients.

    Returns:
    -------
        New tensor.

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data `ls` and shape `shape`.

    Args:
    ----
        ls: Data for tensor.
        shape: Shape of tensor.
        backend: Tensor backend.
        requires_grad: Whether the tensor requires gradients.

    Returns:
    -------
        New tensor.

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape inferred from `ls`.

    Args:
    ----
        ls: Data for tensor.
        backend: Tensor backend.
        requires_grad: Whether the tensor requires gradients.

    Returns:
    -------
        New tensor.

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors
def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the central difference for a gradient.

    Args:
    ----
        f: Function for which to compute the gradient.
        vals: Tensors to differentiate.
        arg: Index of the argument to differentiate with respect to.
        epsilon: Perturbation for finite difference.
        ind: Index to perturb.

    Returns:
    -------
        Approximate gradient.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check the gradient of a function using central differences.

    Args:
    ----
        f: Function for which to check gradients.
        vals: Tensors to differentiate.

    Raises:
    ------
        AssertionError if the gradient check fails.

    """
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """
    Gradient check error for function %s.
    Input %s
    Received derivative %f for argument %d and index %s,
    but was expecting derivative %f from central difference.
    """
    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
