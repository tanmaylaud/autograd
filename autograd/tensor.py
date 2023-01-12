from typing import List, NamedTuple, Callable, Optional, Union
import numpy as np

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

Arrayable = Union[float, list, np.ndarray]

def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)

Tensorable = Union['Tensor', float, np.ndarray]

def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


class Tensor:
    def __init__(self,
                 data: Arrayable,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = None) -> None:
        self._data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self._data.shape
        self.grad: Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = new_data
        # Setting the data manually means we invalidate the gradient.
        self.grad = None

    def reshape(self, *shape):
        self.data = self._data.reshape(*shape)
        return self

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other) -> 'Tensor':
        """gets called if I do t + other"""
        from autograd.tensor_ops import _add
        return _add(self, ensure_tensor(other))

    def __radd__(self, other) -> 'Tensor':
        """gets called if I do other + t"""
        from autograd.tensor_ops import _add
        return _add(ensure_tensor(other), self)

    def __iadd__(self, other) -> 'Tensor':
        """when we do t += other"""
        self.data = self.data + ensure_tensor(other).data
        return self

    def __isub__(self, other) -> 'Tensor':
        """when we do t -= other"""
        self.data = self.data - ensure_tensor(other).data
        return self

    def __imul__(self, other) -> 'Tensor':
        """when we do t *= other"""
        self.data = self.data * ensure_tensor(other).data
        return self

    def __mul__(self, other) -> 'Tensor':
        from autograd.tensor_ops import _mul
        return _mul(self, ensure_tensor(other))

    def __rmul__(self, other) -> 'Tensor':
        from autograd.tensor_ops import _mul
        return _mul(ensure_tensor(other), self)

    def __matmul__(self, other) -> 'Tensor':
        from autograd.tensor_ops import _matmul
        return _matmul(self, other)

    def __neg__(self) -> 'Tensor':
        from autograd.tensor_ops import _neg
        return _neg(self)

    def __sub__(self, other) -> 'Tensor':
        from autograd.tensor_ops import _sub
        return _sub(self, ensure_tensor(other))

    def __rsub__(self, other) -> 'Tensor':
        from autograd.tensor_ops import _sub
        return _sub(ensure_tensor(other), self)

    def __getitem__(self, idxs) -> 'Tensor':
        from autograd.tensor_ops import _slice
        return _slice(self, idxs)

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        self.grad.data = self.grad.data + grad.data  # type: ignore

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self) -> 'Tensor':
        return tensor_sum(self)


def tensor_sum(t: Tensor) -> Tensor:
    """
    Takes a tensor and returns the 0-tensor
    that's the sum of all its elements.
    """
    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily a 0-tensor, so each input element
            contributes that much
            """
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]

    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)