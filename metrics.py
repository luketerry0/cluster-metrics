import torch.distributed as dist
import os
import math
import torch

def inertia(centroids, clusters):
    inertias = torch.zeros(len(centroids))

    # calculate the inertias for nodes which belong to this rank
    n_clusters_per_rank = math.ceil(len(centroids)/dist.get_world_size())
    for idx in range(dist.get_rank()*n_clusters_per_rank, (dist.get_rank()*n_clusters_per_rank)+n_clusters_per_rank):
        inertia=torch.mean(torch.square(torch.sub(centroids[idx],clusters[idx])))
        inertias[idx] = inertia
    print(inertias)

