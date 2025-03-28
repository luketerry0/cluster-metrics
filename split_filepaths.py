"""
this file just reads in the filepaths key, and saves on object containing only filepaths for the first shard
edited slightly to provide a super small debug set...
"""

import pickle
import numpy as np

KEY_PATH = '/ourdisk/hpc/ai2es/jroth/400M_samples_list.pkl'
SHARD_PATHS = ['/ourdisk/hpc/ai2es/jroth/Meta-Co-Training/data_processing/LAION-embed-DINOv2/shard_0',
'/ourdisk/hpc/ai2es/jroth/Meta-Co-Training/data_processing/LAION-embed-DINOv2/shard_1',
'/ourdisk/hpc/ai2es/jroth/Meta-Co-Training/data_processing/LAION-embed-DINOv2/shard_2']
OUTPUT_PATH = '/ourdisk/hpc/ai2es/luketerry/npy_laion_embeddings/filenames_shard_0-2_set.pkl'
DATA_OUTPUT_PATH = '/ourdisk/hpc/ai2es/luketerry/npy_laion_embeddings/laion_embeddings_shard_0-2.npy'

data = []


for path in SHARD_PATHS:
    data.append(np.load(path, allow_pickle=True)[0])

full_data = np.concatenate(data)
np.save(DATA_OUTPUT_PATH, full_data)
with open(KEY_PATH, "rb") as fp:
    filepaths = pickle.load(fp)
    first_shard_filepaths = filepaths

with open(OUTPUT_PATH, 'wb') as ofp:
    pickle.dump(first_shard_filepaths, ofp)

    print(first_shard_filepaths)
    print('complete')