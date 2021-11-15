from pathlib import Path
import sys

import sgkit as sg
import zarr


THRESHOLD = 0.2
DEFAULT_WINDOW_SIZE = 500

def main():
    zarr_path = sys.argv[1]
    base_name = Path(zarr_path).stem

    if len(sys.argv) > 2:
        window_size = int(sys.argv[2])
    else:
        window_size = DEFAULT_WINDOW_SIZE

    ds = sg.load_dataset(zarr_path)
    if "dosage" not in ds.variables:
        ds["dosage"] = ds.call_genotype.sum("ploidy")

    ds = sg.window_by_variant(ds, size=window_size, merge=True)
    pruned_ds = sg.ld_prune(ds, threshold=THRESHOLD)
    pruned_ds = pruned_ds.unify_chunks()

    pruned_ds.to_zarr(base_name + ".pruned_at_{}.zarr".format(THRESHOLD), "w")


if __name__ == "__main__":
    main()
