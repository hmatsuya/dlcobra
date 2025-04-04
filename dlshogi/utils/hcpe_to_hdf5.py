import os  # Import os to get file sizes
import cshogi
import numpy as np
import argparse
import dask.array as da
from dask import delayed

parser = argparse.ArgumentParser()
parser.add_argument('out_file')
parser.add_argument('in_files', nargs='+')
parser.add_argument('--shuffle', action='store_true', help='Shuffle the data before saving')
parser.add_argument('--chunk', type=int, default=1000000, help='Chunk size', dest='chunk')
args = parser.parse_args()

# Get the size of the data type
dtype = cshogi.HuffmanCodedPosAndEval
dtype_size = np.dtype(dtype).itemsize

# Calculate the shape for each file dynamically
arrays = []
for file in args.in_files:
    file_size = os.path.getsize(file)  # Get the file size in bytes
    num_elements = file_size // dtype_size  # Calculate the number of elements
    shape_per_file = (num_elements,)  # Assuming 1D shape; adjust if needed
    print(f"{file}: {num_elements} elements", flush=True)
    arrays.append(
        da.from_delayed(
            delayed(np.fromfile)(file, dtype=dtype).reshape(shape_per_file),
            shape=shape_per_file,
            dtype=dtype,
        )
    )

combined = da.concatenate(arrays, axis=0).rechunk((args.chunk,))
print(combined.shape, flush=True)
print(combined.dtype, flush=True)
print(f"chunksize: {combined.chunksize}")

if args.shuffle:
    print("shuffling...", flush=True)
    shuffled_combined = da.random.permutation(combined)
    combined = shuffled_combined

print(f"saving to {args.out_file}", flush=True)
combined.to_hdf5(args.out_file, "/data")
print("done")
