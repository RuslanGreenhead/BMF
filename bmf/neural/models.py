import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.function  import Function, InplaceFunction

import numpy as np
from typing import Tuple

from .binary_modules import SignActivation, shifted_sigmoid, shifted_scaled_tanh

# ----------------------------------------------- Specific Architectures ----------------------------------------------- #

class NeuralBMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embedding_b = bm.BinarizedEmbedding(n_users, embedding_dim)
        self.item_embedding_b = bm.BinarizedEmbedding(n_items, embedding_dim)

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
        return bm.shifted_sigmoid(res)
    
    def get_factors(self, raw=False):
        '''
        Extracts current (binary / raw) factor-matrices for users and for items
        '''
        with torch.no_grad():
            user_raw = self.user_embedding_b.weight.detach().clone().numpy()
            item_raw = self.item_embedding_b.weight.detach().clone().numpy()
            
            if raw:
                return user_raw, item_raw

            user_bin = (np.sign(user_raw) + 1) / 2
            item_bin = (np.sign(item_raw) + 1) / 2

            return user_bin, item_bin
        

    def _init_weight(self):
        nn.init.xavier_uniform_(self.user_embedding_b.weight)     # default distribution - normal
        nn.init.xavier_uniform_(self.item_embedding_b.weight)


class NeuralBMF_1(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.user_linear_1 = nn.Linear(embedding_dim, embedding_dim)
        self.user_linear_2 = nn.Linear(embedding_dim, embedding_dim)
        self.item_linear_1 = nn.Linear(embedding_dim, embedding_dim)
        self.item_linear_2 = nn.Linear(embedding_dim, embedding_dim)

        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (batch_size, 3)
        users = x[:, 0]    # -> (b_size)
        items = x[:, 1]    # -> (b_size)

        user_vectors = self.sigm(self.user_linear_1(self.user_embedding(users)))    # -> (b_size, emb_dim)
        user_vectors = self.tanh(self.user_linear_2(user_vectors))                  # -> (b_size, emb_dim)
        item_vectors = self.sigm(self.item_linear_1(self.item_embedding(items)))    # -> (b_size, emb_dim)
        item_vectors = self.tanh(self.item_linear_2(item_vectors))                  # -> (b_size, emb_dim)

        # print("user_vectors:", user_vectors)
        # print("item_vectors:", item_vectors)

        res = torch.sum(torch.mul(user_vectors, item_vectors), axis=1)    # -> (b_size)
        return bm.shifted_sigmoid(res)
    
    def get_factors(self, raw=False):
        '''
        Extracts current (binary / raw) factor-matrices for users and for items
        '''
        with torch.no_grad():
            user_raw = self.sigm(self.user_linear_1(self.user_embedding.weight))
            user_raw = self.tanh(self.user_linear_2(user_raw))
            item_raw = self.sigm(self.item_linear_1(self.item_embedding.weight))
            item_raw = self.tanh(self.item_linear_2(item_raw))

            if raw:
                return user_raw.detach().clone().numpy(), item_raw.detach().clone().numpy()
            
            user_bin = (np.sign(user_raw.detach().clone().numpy()) + 1) / 2
            item_bin = (np.sign(item_raw.detach().clone().numpy()) + 1) / 2

            return user_bin, item_bin

    def _init_weight(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)     # default distribution - normal
        nn.init.xavier_uniform_(self.item_embedding.weight)



# ----------------------------------------------- General Architectures ----------------------------------------------- #
class NeuralBMF_large(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int,
                 hidden_dim: int, output_act: str, init_weights: bool, **kwargs) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.user_linear_1 = nn.Linear(embedding_dim, hidden_dim)
        self.user_linear_2 = nn.Linear(hidden_dim, embedding_dim)
        self.item_linear_1 = nn.Linear(embedding_dim, hidden_dim)
        self.item_linear_2 = nn.Linear(hidden_dim, embedding_dim)

        self.sigm = nn.Sigmoid()
        self.pooling = nn.MaxPool1d(kernel_size=embedding_dim)
        self.output_act = output_act
        self.kwargs = kwargs
        
        if init_weights: self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (batch_size, 3)
        users = x[:, 0]    # -> (b_size)
        items = x[:, 1]    # -> (b_size)

        user_vectors = self.sigm(self.user_linear_1(self.user_embedding(users)))    # -> (b_size, emb_dim)
        user_vectors = (binarized(self.user_linear_2(user_vectors)) + 1) / 2        # -> (b_size, emb_dim)
        item_vectors = self.sigm(self.item_linear_1(self.item_embedding(items)))    # -> (b_size, emb_dim)
        item_vectors = (binarized(self.item_linear_2(item_vectors)) + 1) / 2        # -> (b_size, emb_dim)

        if self.output_act == "shifted_sigmoid":
            res = torch.sum(torch.mul(user_vectors, item_vectors), axis=1)             # -> (b_size)
            res = bm.shifted_sigmoid(res)
        elif self.output_act == "tanh":
            res = torch.sum(torch.mul(user_vectors, item_vectors), axis=1)             # -> (b_size)
            res = torch.tanh(self.kwargs["tanh_coef"] * res)
        elif self.output_act == "maxpool":
            res = self.pooling(torch.mul(user_vectors, item_vectors)).squeeze(1)       # -> (b_size)
            
        return res
    
    def get_factors(self, raw=False):
        '''
        Extracts current (binary / raw) factor-matrices for users and for items
        '''
        with torch.no_grad():
            user_raw = self.sigm(self.user_linear_1(self.user_embedding.weight))
            user_raw = self.user_linear_2(user_raw)
            item_raw = self.sigm(self.item_linear_1(self.item_embedding.weight))
            item_raw = self.item_linear_2(item_raw)

            if raw:
                return user_raw.detach().clone().numpy(), item_raw.detach().clone().numpy()
            
            user_bin = (np.sign(user_raw.detach().clone().numpy()) + 1) / 2
            item_bin = (np.sign(item_raw.detach().clone().numpy()) + 1) / 2

            return user_bin, item_bin

    def _init_weight(self):
        nn.init.xavier_uniform_(self.user_linear_1.weight)     # default distribution - normal
        nn.init.xavier_uniform_(self.user_linear_2.weight)
        nn.init.xavier_uniform_(self.item_linear_1.weight)
        nn.init.xavier_uniform_(self.item_linear_2.weight)