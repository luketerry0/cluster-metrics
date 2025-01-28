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
from metrics import inertia, simplified_silhouette, db_index, silhouette_coef
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
print(f"Inertias Match: {(float(sum(my_inertia)) - sklearn_inertia) < CALCULATION_THRESHOLD}")

# compare silhouette score
sklearn_silhouette = silhouette_score(data, labels, metric='euclidean')
my_silhouette = silhouette_coef(centroid_dist_format, label_dist_format)
print(sklearn_silhouette)
print(float(sum(my_silhouette)/len(my_silhouette)))

# compare simplified silhouette
my_simplified_silhouette = simplified_silhouette(centroid_dist_format, label_dist_format, coeffiecient=False)
# scikit-learn doesn't have this, so I have some super simple code here
distances_to_centroids = torch.sort(torch.cdist(data.double(), centroid_dist_format)).values
# distances to centroids is now a sorted list of each datapoints distance to the centroids (sorted by dist)
# smallest distance is the centroid of the datapoint's cluster, and the second smallest is the minimum distance to a different centroid
# a' and b' from https://en.wikipedia.org/wiki/Silhouette_(clustering)
trimmed_distance = distances_to_centroids[:, :2]
denominator = torch.max(trimmed_distance, dim=1).values
numerator = torch.sub(trimmed_distance[:, 1], trimmed_distance[:, 0])
simplified_silhouettes = torch.tensor(numerator/denominator)
# now just take the average for each cluster
silhouettes_by_cluster = [[] for i in range(NUM_CLUSTERS)]
for i in range(NUM_DATAPOINTS):
    silhouettes_by_cluster[labels[i]].append(float(simplified_silhouettes[i]))
silhouettes_by_cluster = [sum(c)/len(c) for c in silhouettes_by_cluster]
print(silhouettes_by_cluster)
print(my_simplified_silhouette)



dist.destroy_process_group()

