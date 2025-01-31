"""
this file compares my implementation against scikit learn as a sanity check

run with run_sanity_check_local.sh
"""

from sklearn.cluster import k_means
from sklearn.metrics import silhouette_score
from torch.nn import PairwiseDistance
import numpy as np
import torch
import torch.distributed as dist
from metrics import inertia, simplified_silhouette, db_index, silhouette_coef, validation_simple_silhouettes
import os
import math

NUM_CLUSTERS = 5
NUM_DATAPOINTS = 10
DATA_DIMENSION = 10
CALCULATION_THRESHOLD = 0.000001

# for testing reproducibility
torch.manual_seed(0)

# pytorch process group initialization
world_size = int(os.environ["WORLD_SIZE"])
# the 'rank' of the current process, a unique id for the current gpu
rank = int(os.environ["RANK"])
torch.cuda.set_device(rank % torch.cuda.device_count())
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# create clusters of dummy data
data =  torch.rand(NUM_DATAPOINTS, DATA_DIMENSION)
centroids, labels, sklearn_inertia = k_means(data, n_clusters=NUM_CLUSTERS, random_state=0, n_init="auto")

# convert centroids into format used by distributed code
centroid_dist_format = torch.tensor(centroids)

# convert labels into format used by distributed code
label_dist_format = []
for i in range(NUM_CLUSTERS):
    label_dist_format.append(data[labels == i])


# compare inertia
my_inertia = inertia(centroid_dist_format, label_dist_format)
# print(f"Inertias Match: {(float(sum(my_inertia)) - sklearn_inertia) < CALCULATION_THRESHOLD}")

# compare silhouette score
sklearn_silhouette = silhouette_score(data, labels, metric='euclidean')
my_silhouette = silhouette_coef(centroid_dist_format, label_dist_format)
# print(sklearn_silhouette)
# print(float(sum(my_silhouette)/len(my_silhouette)))

# compare simplified silhouette
my_simple_silhouettes = simplified_silhouette(centroid_dist_format, label_dist_format)
validation_simple_silhouettes = validation_simple_silhouettes(centroid_dist_format, label_dist_format)

print(my_simple_silhouettes)
print(validation_simple_silhouettes)

dist.destroy_process_group()

