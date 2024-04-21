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



# ----------------------------------------------- Data loading + processing ---------------------------------------------- #

# ------------------------------------------------------- MOVIELENS ------------------------------------------------------ #

def generate_dataset(path, variant="20m", outputpath="."):
    """Generates a hdf5 movielens datasetfile from the raw datafiles found at:
    https://grouplens.org/datasets/movielens/20m/

    You shouldn't have to run this yourself, and can instead just download the
    output using the 'get_movielens' function./
    """
    filename = os.path.join(outputpath, f"movielens_{variant}.hdf5")

    if variant == "20m":
        ratings, movies = _read_dataframes_20M(path)
    elif variant == "100k":
        ratings, movies = _read_dataframes_100k(path)
    else:
        ratings, movies = _read_dataframes(path)

    _hfd5_from_dataframe(ratings, movies, filename)


def _read_dataframes_20M(path):
    """reads in the movielens 20M"""
    import pandas

    ratings = pandas.read_csv(os.path.join(path, "ratings.csv"))
    movies = pandas.read_csv(os.path.join(path, "movies.csv"))

    return ratings, movies


def _read_dataframes_100k(path):
    """reads in the movielens 100k dataset"""
    import pandas

    ratings = pandas.read_table(
        os.path.join(path, "u.data"), names=["userId", "movieId", "rating", "timestamp"]
    )

    movies = pandas.read_csv(
        os.path.join(path, "u.item"),
        names=["movieId", "title"],
        usecols=[0, 1],
        delimiter="|",
        encoding="ISO-8859-1",
    )

    return ratings, movies


def _read_dataframes(path):
    import pandas

    ratings = pandas.read_csv(
        os.path.join(path, "ratings.dat"),
        delimiter="::",
        names=["userId", "movieId", "rating", "timestamp"],
    )

    movies = pandas.read_table(
        os.path.join(path, "movies.dat"), delimiter="::", names=["movieId", "title", "genres"]
    )
    return ratings, movies


def _hfd5_from_dataframe(ratings, movies, outputfilename):
    # transform ratings dataframe into a sparse matrix
    m = coo_matrix(
        (ratings["rating"].astype(np.float32), (ratings["movieId"], ratings["userId"]))
    ).tocsr()

    with h5py.File(outputfilename, "w") as f:
        # write out the ratings matrix
        g = f.create_group("movie_user_ratings")
        g.create_dataset("data", data=m.data)
        g.create_dataset("indptr", data=m.indptr)
        g.create_dataset("indices", data=m.indices)

        # write out the titles as a numpy array
        titles = np.empty(shape=(movies.movieId.max() + 1,), dtype=np.object)
        titles[movies.movieId] = movies.title
        dt = h5py.special_dtype(vlen=str)
        dset = f.create_dataset("movie", (len(titles),), dtype=dt)
        dset[:] = titles