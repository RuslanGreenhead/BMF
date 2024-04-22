import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.function  import Function, InplaceFunction

import numpy as np
from typing import Tuple, Dict, Any, List
from abc import ABCMeta, abstractmethod
from functools import partial

# ---------------------------------------------------- Basic tools ---------------------------------------------------- # 

class Binarize(InplaceFunction):
    def forward(ctx, input, quant_mode='det', allow_scale=False, inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        scale = output.abs().max() if allow_scale else 1

        if quant_mode == 'det':
            return output.div(scale).sign().mul(scale)
        else:
            return output.div(scale).add_(1).div_(2).add_(torch.rand(output.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1).mul(scale)

    def backward(ctx, grad_output):
        # STE
        grad_input = grad_output
        return grad_input, None, None, None


def binarized(input, quant_mode='det'):
    return Binarize.apply(input, quant_mode)


class BinarizedLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:      # for ResNet case
            input_b=binarized(input)
        weight_b=binarized(self.weight)
        out = nn.functional.linear(input_b,weight_b)

        # if not self.bias is None:                              # adding bias
        #     self.bias.org=self.bias.data.clone()
        #     out += self.bias.view(1, -1).expand_as(out)

        return out


class BinarizedEmbedding(nn.Embedding):
    def __init__(self, *kargs, **kwargs):
        super(BinarizedEmbedding, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        weight_b = binarized(self.weight)
        output = nn.functional.embedding(input, weight_b)

        return output


def shifted_sigmoid(x):
    return torch.sigmoid(x - 1)

def shifted_scaled_tanh(x, coef):                   
    return 0.5 * (torch.tanh(coef * (x - 1)) + 1)



# ---------------------------------------------------- Advanced tools ---------------------------------------------------- #

def _with_args(cls_or_self: Any, **kwargs: Dict[str, Any]) -> Any:
    r'''Wrapper that allows creation of class factories.
    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances.
    Source: https://github.com/pytorch/pytorch/blob/b02c932fb67717cb26d6258908541b670faa4e72/torch/quantization/observer.py
    Example::
        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
        >>> foo_instance1 = foo_builder()
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1) == id(foo_instance2)
        False
    '''

    class _PartialWrapper(object):
        def __init__(self, p):
            self.p = p

        def __call__(self, *args, **keywords):
            return self.p(*args, **keywords)

        def __repr__(self):
            return self.p.__repr__()

        with_args = _with_args
    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r


ABC = ABCMeta(str("ABC"), (object,), {})


class BinarizerBase(ABC, nn.Module):
    def __init__(self) -> None:
        super(BinarizerBase, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    with_args = classmethod(_with_args)


class SignActivation(Function):
    r'''Applies the sign function element-wise
    :math:`\text{sgn(x)} = \begin{cases} -1 & \text{if } x < 0, \\ 1 & \text{if} x >0  \end{cases}`
    the gradients of which are computed using a STE, namely using :math:`\text{hardtanh(x)}`.
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> input = torch.randn(3)
        >>> output = SignActivation.apply(input)
    '''

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return input.sign()

        # x = torch.tanh(input * 3)       # some alternative way  -> in this case we have to DELETE 
        # with torch.no_grad():           # the following backward
        #     x = torch.sign(x)
        # return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(input.ge(1) | input.le(-1), 0)       # clipping elements
        return grad_input


class SignActivationStochastic(SignActivation):
    r'''Binarize the data using a stochastic binarizer
    :math:`\text{sgn(x)} = \begin{cases} -1 & \text{with probablity } p = \sigma(x), \\ 1 & \text{with probablity } 1 - p \end{cases}`
    the gradients of which are computed using a STE, namely using :math:`\text{hardtanh(x)}`.
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> input = torch.randn(3)
        >>> output = SignActivationStochastic.apply(input)
    '''

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        noise = torch.rand_like(input).sub_(0.5)
        return input.add_(1).div_(2).add_(noise).clamp_(0, 1).round_().mul_(2).sub_(1)


class TanhBinaryActivation(Function):
    r'''Advanced binarization'''

    @staticmethod
    def forward(ctx, input: torch.Tensor, k, t) -> torch.Tensor:
        ctx.save_for_backward(input, k, t)

        return input.sign()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(t * input), 2)) * grad_output  # using derivative of tanh

        return grad_input, None, None
        


class BasicInputBinarizer(BinarizerBase):
    r'''pplies the sign function element-wise.
    nn.Module version of SignActivation.
    '''

    def __init__(self):
        super(BasicInputBinarizer, self).__init__()

    def forward(self, x: torch.Tensor) -> None:
        return SignActivation.apply(x)


class StochasticInputBinarizer(BinarizerBase):
    r'''Applies the sign function element-wise.
    nn.Module version of SignActivation.
    '''

    def __init__(self):
        super(StochasticInputBinarizer, self).__init__()

    def forward(self, x: torch.Tensor):
        return SignActivationStochastic.apply(x)


class TanhInputBinarizer(BinarizerBase):
    r'''Applies STE with scaled tanh under the hood.
    nn.Module version of TanhBinaryActivation.
    '''

    def __init__(self, k, t):
        super(TanhInputBinarizer, self).__init__()
        self.k = torch.tensor([k]).float()
        self.t = torch.tensor([t]).float()

    def forward(self, x: torch.Tensor):
        return TanhBinaryActivation.apply(x, self.k, self.t)
