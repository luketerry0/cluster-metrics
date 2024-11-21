import torch.distributed as dist
import os
import math
import torch
from tqdm import tqdm

def inertia(centroids, clusters):
    inertias = torch.zeros(len(centroids)).to(device="cuda")

    # calculate the inertias for nodes which belong to this rank
    print("calculating inertias")
    rank = dist.get_rank()
    for idx in tqdm(range(rank, len(centroids), dist.get_world_size())):
        centroid = torch.unsqueeze(centroids[idx], 0).to(device="cuda")
        cluster = clusters[idx].to(device="cuda").double()
        inertia = torch.mean(torch.square(torch.cdist(centroid, cluster)))

        inertias[idx] = inertia

        # gather up the tensors
        if rank == 0:
            dist.gather(inertias[idx], [inertias[a] for a in range(idx, idx+dist.get_world_size())])
        else:
            dist.gather(inertias[idx])


        # clean up
        del centroid
        del cluster
        del inertia
    
    return inertias

# # see https://en.wikipedia.org/wiki/Silhouette_(clustering) for a/b notation
# def silhouette_coef(centroids, clusters):
#     mean_si_over_clusters = torch.zeros(len(centroids))
#     for cluster_idx in range(len(centroids)):
#         # information about the current cluster
#         curr_cluster = clusters[cluster_idx]

#         # initialize tensors to hold a and b values
#         a_values = torch.zeros(len(curr_cluster))
#         b_values = torch.zeros(len(curr_cluster))

#         if (len(curr_cluster) > 1):

#             for point_idx in range(dist.get_rank(), len(curr_cluster), dist.get_world_size()):
#                 curr_point = torch.unsqueeze(curr_cluster[point_idx], 0)

#                 # calculate the "a" value
#                 cluster_points_excluding_point = torch.cat((curr_cluster[:point_idx], curr_cluster[point_idx + 1:]))
#                 a = torch.mean(torch.cdist(curr_point, cluster_points_excluding_point))
#                 a_values[point_idx] = a


#                 # calculate the "b" value
#                 b = torch.inf
#                 other_clusters = clusters[:cluster_idx]
#                 other_clusters.extend(clusters[cluster_idx + 1:])
#                 for other_cluster in other_clusters:
#                     avg_distance = torch.mean(torch.cdist(curr_point, other_cluster))
#                     if avg_distance < b:
#                         b = avg_distance
#                 b_values[point_idx] = b
                    

#             # get a and b values
#             dist.reduce(a_values, 0, torch.distributed.ReduceOp.MAX)
#             dist.reduce(b_values, 0, torch.distributed.ReduceOp.MAX)

#             # calculate s_i values
#             max_a_or_b = torch.max(torch.cat((a_values,b_values)))
#             s_i_values = (b_values - a_values)/max_a_or_b

#             # calculate mean s_i value for this cluster
#             mean_s_i = torch.mean(s_i_values)

#             mean_si_over_clusters[cluster_idx] = mean_s_i
        

#     silhouette_coef = max(mean_si_over_clusters)
#     return silhouette_coef


def simplified_silhouette(centroids, clusters):

    silhouette_coef = torch.zeros(1).to(device="cuda")
    print("calculating silhouettes")
    for cluster_idx in tqdm(range(dist.get_rank(), len(centroids), dist.get_world_size())):
        centroid = torch.unsqueeze(centroids[cluster_idx], 0).double().to(device="cuda")
        curr_cluster = clusters[cluster_idx].to(device="cuda").double()

        # calculate a' values
        a = torch.cdist(curr_cluster, centroid).to(device="cuda")
        del centroid

        # calculate b' values
        block_size = 100
        other_centroids = torch.cat((centroids[:cluster_idx], centroids[cluster_idx + 1:])).to(device="cuda")
        if (len(curr_cluster) > block_size):
            # if the GPU can't handle a big cluster, process it in blocks
            running_pairwise_distances = torch.tensor([]).to(device="cpu")
            for idx in range(0, len(curr_cluster) + 1, block_size):
                bottom_of_range = idx
                top_of_range = min(idx + block_size, len(curr_cluster))

                curr_cluster_block = curr_cluster[bottom_of_range:top_of_range]

                pairwise_distances = torch.cdist(curr_cluster_block, other_centroids).to(device="cpu")
                running_pairwise_distances = torch.cat((running_pairwise_distances, pairwise_distances))
                del curr_cluster_block
            running_pairwise_distances.to(device="cuda")
            b = torch.min(running_pairwise_distances, dim=1, keepdim=True).values.to(device="cuda")
            del running_pairwise_distances
        else:
            pairwise_distances = torch.cdist(curr_cluster, other_centroids)
            b = torch.min(pairwise_distances, dim=1, keepdim=True).values

        # clean up a lil bit
        del curr_cluster
        del other_centroids


        # calculate average s value for this cluster
        s_values = (b - a)/float(max(torch.cat((a, b))))
        curr_sill_val = sum(s_values)/len(s_values)
        if curr_sill_val > silhouette_coef:
            silhouette_coef = curr_sill_val
        
        # clean up again
        del a
        del b
        del curr_sill_val

    # gather up the max silhouette coef
    dist.reduce(silhouette_coef, 0, dist.ReduceOp.MAX)

    return silhouette_coef


    
    
