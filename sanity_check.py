"""
this file compares my implementation against scikit learn as a sanity check

run with run_sanity_check_local.sh
"""

from sklearn.cluster import k_means
from sklearn.metrics import silhouette_score, davies_bouldin_score
from torch.nn import PairwiseDistance
import numpy as np
import torch
import torch.distributed as dist
from metrics import inertia, simplified_silhouette, db_index, silhouette_coef, validation_simple_silhouettes, MetricsCalculator
import os
import math

NUM_CLUSTERS = 18
NUM_DATAPOINTS = 100
DATA_DIMENSION = 787
CALCULATION_THRESHOLD = 0.000001

# for testing reproducibility
# torch.manual_seed(0)

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

# convert cluster assignment to a form the calculator expects
assignment_dist_format = []
for i in range(NUM_CLUSTERS):
    curr_assignment = [i for i, x in enumerate(labels == i) if x]
    assignment_dist_format.append(np.array(curr_assignment))
metrics_calulator = MetricsCalculator(centroid_dist_format, label_dist_format, DATA_DIMENSION, assignment_dist_format)


# compare inertia
my_inertia = metrics_calulator.inertia()
print(f"Inertias Match: {(float(sum(my_inertia)) - sklearn_inertia) < CALCULATION_THRESHOLD}")

# # compare silhouette score
# sklearn_silhouette = silhouette_score(data, labels, metric='euclidean')
# my_silhouette = silhouette_coef(centroid_dist_format, label_dist_format)

# # compare simplified silhouette
# my_simple_silhouettes = simplified_silhouette(centroid_dist_format, label_dist_format)
# validation_simple_silhouettes = validation_simple_silhouettes(centroid_dist_format, label_dist_format)
# print(f"Simplified Silhouettes Match: {((my_simple_silhouettes - validation_simple_silhouettes) < CALCULATION_THRESHOLD).all()}")

# # compare Davies Bouldin index
# my_db_index = db_index(centroid_dist_format, label_dist_format)
# sklearn_db_index = davies_bouldin_score(data, labels)
# print(f"Davies-Bouldin Index Match: {(float(my_db_index) - sklearn_db_index) < CALCULATION_THRESHOLD}")


dist.destroy_process_group()

