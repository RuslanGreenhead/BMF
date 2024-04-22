import torch
import torch.nn as nn
import numpy as np

from typing import Tuple


# -------------------------------------------- NCF implementation -------------------------------------------- #

class GeneralizedMatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        # self.user_embedding = bm.BinarizedEmbedding(n_users, embedding_dim)
        # self.item_embedding = bm.BinarizedEmbedding(n_items, embedding_dim)

        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (batch_size, 3)
        users = x[:, 0]    # -> (b_size)
        items = x[:, 1]    # -> (b_size)

        user_vectors = (self.user_embedding(users) + 1) / 2    # -> (b_size, emb_dim)
        item_vectors = (self.item_embedding(items) + 1) / 2    # -> (b_size, emb_dim)

        return torch.mul(user_vectors, item_vectors)    # -> (b_size, emb_dim)

    def _init_weight(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)     # default distribution - normal
        nn.init.xavier_uniform_(self.item_embedding.weight)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int,
                 mlp_layers: Tuple[int, ...]) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.MLP = self._make_layers(layers=mlp_layers, embedding_dim=embedding_dim)
        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (b_size, 3)
        users = x[:, 0]    # -> (b_size)
        items = x[:, 1]    # -> (b_size)

        user_vectors = self.user_embedding(users)    # -> (b_size, emb_dim)
        item_vectors = self.item_embedding(items)    # -> (b_size, emb_dim)

        concat_vectors = torch.concat((user_vectors, item_vectors), dim=1)    # -> (b_size, 2 * emb_dim)

        return self.MLP(concat_vectors)    # -> (b_size, mlp_layers[-1])

    def _make_layers(self, layers: Tuple[int, ...], embedding_dim: int) -> nn.Module:
        if not layers:
            raise ValueError("should have at least one mlp layer.")

        mlp_layers = nn.Sequential()
        past_dims = embedding_dim * 2

        for layer in layers:
            mlp_layers.append(nn.Linear(past_dims, layer))
            mlp_layers.append(nn.ReLU())
            past_dims = layer

        return mlp_layers

    def _init_weight(self):
        pass


class NeuralMatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, GMF_embedding_dim: int, MLP_embedding_dim: int,
                 mlp_layers: Tuple[int, ...]) -> None:
        super().__init__()
        self.GMF = GeneralizedMatrixFactorization(
            n_users=n_users, n_items=n_items, embedding_dim=GMF_embedding_dim
        )
        self.MLP = MultiLayerPerceptron(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=MLP_embedding_dim,
            mlp_layers=mlp_layers,
        )

        # self.predict_layer = nn.Linear(GMF_embedding_dim + mlp_layers[-1], 1)
        self.predict_layer = nn.Linear(GMF_embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (b_size, 3)
        gmf_vectors = self.GMF(x)    # -> (b_size, emb_dim)
        # mlp_vectors = self.MLP(x)    # -> (b_size, mlp_layers[-1])

        # concat_vectors = torch.concat((gmf_vectors, mlp_vectors), dim=1)    # -> (b_size, emb_dim + mlp_layers[-1])
        output = self.predict_layer(gmf_vectors)                         # -> (b_size, 1)

        return self.sigmoid(output).squeeze()    # -> (b_size)