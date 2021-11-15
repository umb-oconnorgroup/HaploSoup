import os
from collections import Counter
import math
import sys

# prevent sklearn multiprocessing to let dask handle multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import dask.array as da
from numba import njit
from numba.typed import List
import numpy as np
import sgkit as sg
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import xarray as xr
import yaml


PRIME = np.array(16777619, dtype=np.int32)
SEED = np.array(2166136261, dtype=np.int32)
CONFIG_PATH = "config.yaml"


def hash_shingle(shingled_tensor):
    dims = shingled_tensor.shape
    n_bytes = dims[-1]
    shingle_first_tensor = shingled_tensor.reshape(-1, n_bytes).T
    hash_tensor = SEED
    for i in range(n_bytes):
        hash_tensor = np.bitwise_xor(hash_tensor, shingle_first_tensor[i])
        hash_tensor = hash_tensor * PRIME
    return hash_tensor.reshape(dims[:-1])

def pad_window_size(size, shingle_size):
    remainder = size % shingle_size
    if remainder == 0:
        return 0
    else:
        return shingle_size - remainder

def pad_window(window, shingle_size):
    padding = pad_window_size(window.shape[1], shingle_size)
    if padding == 0:
        return window
    else:
        return np.pad(window, ((0, 0), (0, padding)), mode="constant")

def padded_unique(ar):
    unique = np.unique(ar)
    padded_len = math.prod(ar.shape) + 1
    padded = np.pad(unique, (0, padded_len - len(unique)), mode="constant", constant_values=len(unique))
    return np.expand_dims(padded, axis=0)

def convert_to_hash_matrix(hashes):
    unique_hashes, inverse_indices = np.unique(hashes, return_inverse=True)
    inverse_indices = inverse_indices.reshape(hashes.shape)
    hash_matrix = np.zeros((inverse_indices.shape[0], unique_hashes.shape[0]), dtype=np.uint8)
    for i in range(inverse_indices.shape[0]):
        hash_matrix[i][inverse_indices[i]] = 1
    return hash_matrix

@njit
def minhash_signature(hash_matrix, permutations=None, block_id=None):
    permutations = permutations[block_id[1]]
    minhash_signature_matrix = np.empty((hash_matrix.shape[0], permutations.shape[0]), dtype=np.int32)
    minhash_signature_matrix.fill(hash_matrix.shape[1])
    for i in range(hash_matrix.shape[0]):
        for p in range(permutations.shape[0]):
            for j in range(permutations.shape[1]):
                if hash_matrix[i, permutations[p, j]] == 1:
                    minhash_signature_matrix[i, p] = j
                    break
    return minhash_signature_matrix

def select_buckets(lsh_bands, max_components, explained_variance_ratio_target):
    buckets, inverse_indices = np.unique(lsh_bands, return_inverse=True)
    # limit candidate buckets to max_components * 10 most common buckets
    candidate_bucket_indices = np.array([index for index, count in Counter(inverse_indices).most_common(max_components * 10)])
    inverse_indices = inverse_indices.reshape(lsh_bands.shape)
    bucket_matrix = np.zeros((inverse_indices.shape[0], buckets.shape[0]), dtype=np.uint8)
    for i in range(inverse_indices.shape[0]):
        bucket_matrix[i][inverse_indices[i]] = 1
    bucket_matrix = bucket_matrix[:, candidate_bucket_indices]
    pca = PCA(n_components=max_components, svd_solver="randomized")
    pca.fit(bucket_matrix)
    if pca.explained_variance_ratio_.sum() < explained_variance_ratio_target:
        n_components = max_components
    else:
        n_components = np.argmax(pca.explained_variance_ratio_.cumsum() >= explained_variance_ratio_target) + 1
    kmeans = KMeans(n_clusters=n_components)
    distances = kmeans.fit_transform(pca.components_.T)
    selected_candidate_bucket_indices = np.array([distances[:, i].argmin() for i in range(n_components)])
    selected_bucket_matrix = np.zeros((inverse_indices.shape[0], max_components * 2 + 1), dtype=np.int32)
    selected_bucket_matrix[:, :n_components] = bucket_matrix[:, selected_candidate_bucket_indices]
    # store n_components to truncate matrix after dask computation
    selected_bucket_matrix[0, -1] = n_components
    # store selected buckets to apply to new data
    selected_bucket_matrix[0, -1 * n_components - 1: -1] = buckets[candidate_bucket_indices[selected_candidate_bucket_indices]]
    return selected_bucket_matrix

def main():

    zarr_path = sys.argv[1]
    msp_path = sys.argv[2]

    with open(CONFIG_PATH, "r") as f:
        config = yaml.load(f, yaml.UnsafeLoader)

    window_start_positions = []
    window_end_positions = []
    window_sizes = []
    for line_i, line in enumerate(open(msp_path, "r")):
        if line_i >= 2:
            _, start, end, _, _, size, _ = line.split("\t", 6)
            window_start_positions.append(int(start))
            window_end_positions.append(int(end))
            window_sizes.append(int(size))

    ds = sg.load_dataset(zarr_path)
    haplotype = ds.call_genotype.chunk({"variants": tuple(window_sizes), "samples": ds.call_genotype.sizes["samples"]}).stack(haplotype=("samples", "ploidy")).T.data

    # for uneven window sizes, pad before reshaping (or could reshape and/or pad with map blocks)
    padded_window_sizes = [window_size + pad_window_size(window_size, config["shingle_size"]) for window_size in haplotype.chunks[1]]
    padded_haplotype = haplotype.map_blocks(lambda x: pad_window(x, config["shingle_size"]), chunks=(haplotype.chunks[0], tuple(padded_window_sizes)), dtype=np.uint8)
    shingled_haplotype = padded_haplotype.reshape(padded_haplotype.shape[0], -1, config["shingle_size"])
    packed_shingled_haplotype = shingled_haplotype.map_blocks(lambda x: np.packbits(x).reshape(x.shape[:-1] +(x.shape[-1] // 8,)), chunks=(shingled_haplotype.chunks[:-1] + (shingled_haplotype.chunksize[-1] // 8,)), dtype=np.uint8)
    shingle_hashes = packed_shingled_haplotype.map_blocks(hash_shingle, drop_axis=2, dtype=np.int32)
    unique_shingle_hashes = shingle_hashes.map_blocks(padded_unique, chunks=((1,), tuple([shingle_hashes.chunksize[0] * window_size + 1 for window_size in shingle_hashes.chunks[1]])), dtype=np.int32).reshape(-1).compute()
    unique_shingle_hashes = np.split(unique_shingle_hashes, indices_or_sections=np.cumsum([shingle_hashes.chunksize[0] * window_size + 1 for window_size in shingle_hashes.chunks[1]])[:-1], axis=0)
    unique_shingle_hashes = [unique_shingle_hashes_window[:unique_shingle_hashes_window[-1]] for unique_shingle_hashes_window in unique_shingle_hashes]
    np.savez("shingle_hashes.npz", *unique_shingle_hashes)
    hash_matrix = shingle_hashes.map_blocks(convert_to_hash_matrix, dtype=np.uint8, chunks=(shingle_hashes.chunksize[0], tuple([len(ar) for ar in unique_shingle_hashes])))
    permutations = [np.stack([np.random.permutation(len(ar)).astype(np.int32) for i in range(config["n_permutations"])]) for ar in unique_shingle_hashes]
    np.savez("permutations.npz", *permutations)
    minhash_signature_matrix = hash_matrix.map_blocks(minhash_signature, dtype=np.int32, chunks=(shingle_hashes.chunksize[0], config["n_permutations"]), permutations=List(permutations))
    lsh_bands = minhash_signature_matrix.map_blocks(lambda x: hash_shingle(x.view(np.uint8).reshape(x.shape[0], config["n_bands"], -1)), chunks=(minhash_signature_matrix.chunksize[0], config["n_bands"]), dtype=np.int32)
    bucket_matrix = lsh_bands.map_blocks(select_buckets, chunks=(lsh_bands.chunksize[0], config["max_components"] * 2 + 1), dtype=np.int32, max_components=config["max_components"], explained_variance_ratio_target=config["explained_variance_ratio_target"]).compute()
    bucket_matrices = np.array_split(bucket_matrix, lsh_bands.npartitions, axis=1)
    haplosoup_window_sizes = [bucket_matrix[0, -1] for bucket_matrix in bucket_matrices]
    buckets = [bucket_matrix[0, -1 * haplosoup_window_size - 1:-1] for bucket_matrix, haplosoup_window_size in zip(bucket_matrices, haplosoup_window_sizes)]
    np.savez("buckets.npz", *buckets)
    bucket_matrices = [bucket_matrix[:, :haplosoup_window_size].astype(np.uint8) for bucket_matrix, haplosoup_window_size in zip(bucket_matrices, haplosoup_window_sizes)]
    bucket_matrix = np.hstack(bucket_matrices)
    haplosoup_embedding = bucket_matrix.T.reshape(bucket_matrix.shape[-1], ds.call_genotype.sizes["samples"], ds.call_genotype.sizes["ploidy"])
    haplosoup_embedding = xr.DataArray(data=haplosoup_embedding, dims=["buckets", "samples", "ploidy"]).sum("ploidy")
    haplosoup_embedding.to_dataset(name="haplosoup_embedding").to_zarr("haplosoup.zarr")

if __name__ == "__main__":
    main()