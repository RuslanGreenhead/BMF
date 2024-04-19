import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


# Binary tensor creator
def get_binary_tensor(size: Tuple, dtype=torch.int8, seed=None):
    return torch.randint(0, 2, size, dtype=dtype)

# Binary patterned matrix creator
def get_pattern_matrix(size: Tuple, rows=[], cols=[], dtype=torch.int8):
    res = torch.zeros(*size)
    for r in rows:
        res[r, :] = torch.ones(size[1])
    for c in cols:
        res[:, c] = torch.ones(size[0])

    return res


# Boolean matrix multiplication #1
def bin_matmul_1(A: np.array, B: np.array):
    return (np.matmul(A, B) > 0).astype(int)


# Boolean matrix multiplication #2
def bin_matmul_2(A: np.array, B: np.array):
    return (A.astype(bool) @ B.astype(bool)).astype(int)

 
# Boolean matrix multiplication #3
def bin_matmul_3(A: np.array, B: np.array):
    res = np.empty((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            res[i, j] = np.any(A[i] * B[:, j])

    return res


# Convert matrix to table with cell ids and cell value
def matrix_to_ids(m: torch.Tensor):
    res = torch.empty(m.numel(), 3, dtype=torch.int64)

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            res[m.shape[1] * i + j] = torch.tensor([i, j, m[i, j]], dtype=torch.int64)

    return res