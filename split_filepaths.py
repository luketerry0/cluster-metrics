"""
this file just reads in the filepaths key, and saves on object containing only filepaths for the first shard
edited slightly to provide a super small debug set...
"""

import pickle
import numpy as np

KEY_PATH = '/ourdisk/hpc/ai2es/jroth/400M_samples_list.pkl'
SHARD_PATH = '/ourdisk/hpc/ai2es/luketerry/npy_laion_embeddings/laion_embeddings_shard_0.npy'
OUTPUT_PATH = '/ourdisk/hpc/ai2es/luketerry/npy_laion_embeddings/filenames_shard_0_debug_set.pkl'
DATA_OUTPUT_PATH = '/ourdisk/hpc/ai2es/luketerry/npy_laion_embeddings/laion_embeddings_debug.npy'

data = np.load(SHARD_PATH)
num_datapoints = 100

np.save(DATA_OUTPUT_PATH, data[:100])
with open(KEY_PATH, "rb") as fp:
    filepaths = pickle.load(fp)
    first_shard_filepaths = filepaths[:num_datapoints]

    with open(OUTPUT_PATH, 'wb') as ofp:
        pickle.dump(first_shard_filepaths, ofp)

    print(first_shard_filepaths)
    print('complete')