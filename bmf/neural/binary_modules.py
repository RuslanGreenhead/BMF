import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.function  import Function, InplaceFunction

import numpy as np
from typing import Tuple


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

        if input.size(1) != 784:      # to fix
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


def foo(): pass


def shifted_sigmoid(x):
    return torch.sigmoid(x - 1)


class NeuralBMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embedding_b = BinarizedEmbedding(n_users, embedding_dim)
        self.item_embedding_b = BinarizedEmbedding(n_items, embedding_dim)

        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (batch_size, 3)
        users = x[:, 0]    # -> (b_size)
        items = x[:, 1]    # -> (b_size)

        user_vectors = (self.user_embedding_b(users) + 1) / 2    # -> (b_size, emb_dim)
        item_vectors = (self.item_embedding_b(items) + 1) / 2    # -> (b_size, emb_dim)

        # print("user_vectors:", user_vectors)
        # print("item_vectors:", item_vectors)

        res = torch.sum(torch.mul(user_vectors, item_vectors), axis=1)    # -> (b_size)
        return shifted_sigmoid(res)

    def _init_weight(self):
        nn.init.xavier_uniform_(self.user_embedding_b.weight)     # default distribution - normal
        nn.init.xavier_uniform_(self.item_embedding_b.weight)