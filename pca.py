from pathlib import Path
import sys

import dask
import dask.array as da
from dask_ml.preprocessing import StandardScaler
# from dask_ml.utils import svd_flip
from joblib import dump
import numpy as np
import sgkit as sg


N_COMPONENTS = 50
N_POWER_ITER = 2

def main():
    zarr_path = sys.argv[1]
    variable_name = sys.argv[2]

    base_name = Path(zarr_path).stem

    ds = sg.load_dataset(zarr_path)
    if variable_name not in ds.variables:
        raise ValueError("{} not in dataset variables".format(variable_name))

    dosage_array = ds[variable_name].T
    standard_scaler = StandardScaler()
    scaled_dosage_array = standard_scaler.fit_transform(dosage_array)
    u, s, vt = da.linalg.svd_compressed(scaled_dosage_array, k=N_COMPONENTS, n_power_iter=N_POWER_ITER, compute=True)
    # next line unnecessary, happens internally in svd_compressed
    #u, vt = svd_flip(u, vt)
    pca = u * s
    explained_variance = (s ** 2) / (dosage_array.shape[0] - 1)
    # the total variance of scaled_dosage_array is the number of variants because of the standard scaling
    explained_variance_ratio = explained_variance / dosage_array.shape[1]
    pca, principal_components, explained_variance_ratio = dask.compute(pca, vt.T, explained_variance_ratio)

    dump(standard_scaler, base_name + ".scaler.joblib")
    np.save(base_name + ".pca.npy", pca)
    np.save(base_name + ".components.npy", principal_components)
    np.save(base_name + ".explained_variance_ratio.npy", explained_variance_ratio)


if __name__ == "__main__":
    main()
