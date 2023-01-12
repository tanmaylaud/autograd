from autograd.module import Module
from autograd.parameter import Parameter


class Linear(Module):
    def __init__(self, in_dim, out_dim, bias=True):
        self.W = Parameter(in_dim, out_dim)
        self.bias = bias
        if bias:
            self.b = Parameter(1, out_dim)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        out = X @ self.W
        if self.bias:
            out = out + self.b
        return out

