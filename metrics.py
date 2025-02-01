import torch.distributed as dist
import os
import math
import torch
from tqdm import tqdm

def print_mem_usage():
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

"""
Calculates the inertia of this clustering

centroids: a 2d tensor of the centroids
clusters: a list of 2d tensors representing the contents of each cluster
"""
def inertia(centroids, clusters):
    inertias = torch.zeros(len(centroids)).to(device="cuda")

    # calculate the inertias for nodes which belong to this rank
    print("calculating inertias")
    rank = dist.get_rank()
    for idx in tqdm(range(rank, len(centroids), dist.get_world_size())):
        centroid = torch.unsqueeze(centroids[idx], 0).to(device="cuda")
        cluster = clusters[idx].to(device="cuda").double()
        inertia = torch.sum(torch.square(torch.cdist(centroid, cluster)))

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
def silhouette_coef(centroids, clusters):
    mean_si_over_clusters = torch.zeros(len(centroids)).to(device="cuda")
    for cluster_idx in range(len(centroids)):
        # information about the current cluster
        curr_cluster = clusters[cluster_idx]

        # initialize tensors to hold a and b values
        a_values = torch.zeros(len(curr_cluster)).to(device="cuda")
        b_values = torch.zeros(len(curr_cluster)).to(device="cuda")

        if (len(curr_cluster) > 1):

            for point_idx in range(dist.get_rank(), len(curr_cluster), dist.get_world_size()):
                curr_point = torch.unsqueeze(curr_cluster[point_idx], 0)

                # calculate the "a" value
                cluster_points_excluding_point = torch.cat((curr_cluster[:point_idx], curr_cluster[point_idx + 1:]))
                a = torch.mean(torch.cdist(curr_point, cluster_points_excluding_point))
                a_values[point_idx] = a


                # calculate the "b" value
                b = torch.inf
                other_clusters = clusters[:cluster_idx]
                other_clusters.extend(clusters[cluster_idx + 1:])
                for other_cluster in other_clusters:
                    avg_distance = torch.mean(torch.cdist(curr_point, other_cluster))
                    if avg_distance < b:
                        b = avg_distance
                b_values[point_idx] = b
                    

            # get a and b values
            dist.reduce(a_values, 0, torch.distributed.ReduceOp.MAX)
            dist.reduce(b_values, 0, torch.distributed.ReduceOp.MAX)

            # calculate s_i values
            max_a_or_b = torch.max(torch.cat((a_values,b_values)))
            s_i_values = (b_values - a_values)/max_a_or_b

            # calculate mean s_i value for this cluster
            mean_s_i = torch.mean(s_i_values)

            mean_si_over_clusters[cluster_idx] = mean_s_i
        

    silhouette_coef = mean_si_over_clusters # = max(mean_si_over_clusters)
    return silhouette_coef

"""
Calculates the silhouette coeffiecient of the clustering

centroids: a 2d tensor of the centroids
clusters: a list of 2d tensors representing the contents of each cluster
if coefficient == True, the silhouette coefficient is returned. Otherwise, a list of each cluster's average silhouette is returned
"""
def simplified_silhouette(centroids, clusters, coeffiecient = True):

    print("calculating silhouettes")
    avg_silhouettes = torch.zeros(len(centroids))
    for cluster_idx in tqdm(range(dist.get_rank(), len(centroids), dist.get_world_size())):
        centroid = torch.unsqueeze(centroids[cluster_idx], 0).double().to(device="cuda")
        curr_cluster = clusters[cluster_idx].to(device="cuda").double()

        # calculate a' values
        a = torch.cdist(curr_cluster, centroid).to(device="cuda")
        del centroid

        # calculate b' values
        block_size = 3000
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
        s_values = torch.div((b - a).t(), torch.max(torch.cat((a, b), dim=1), dim=1).values)
        avg_silhouettes[cluster_idx] = torch.mean(s_values)

        # clean up again
        del a
        del b

    return avg_silhouettes

"""
Calculates the same simplified silhouette as above, but is easier to understand.
Will crash because of cuda memory if ran on large data (this code is for validation only)

centroids and clusters are same parameters as above
"""
def validation_simple_silhouettes(centroids, clusters):
    centroids = centroids
    points = torch.cat(clusters).double()
    cluster_sizes = [a.size(0) for a in clusters]

    # calculate the pairwise distances between each point and every centroid, discarding all except the closest and second closest
    # we're relying here on the fact that in k-means, the closest centroid always corresponds to the point's class
    distances = torch.sort(torch.cdist(points, centroids), dim=1).values[:, :2]
    del points
    silhouettes = (distances[:,1] - distances[:,0]) / torch.max(distances, dim=1).values
    # take the average of each cluster's silhouette 
    cluster_sizes = [a.size(0) for a in clusters]
    avg_silhouettes = torch.zeros(len(cluster_sizes))
    for cluster_idx in range(len(cluster_sizes)):
        mask = torch.cat(
            [torch.zeros(sum(cluster_sizes[:cluster_idx])), 
            torch.ones(cluster_sizes[cluster_idx]), 
            torch.zeros(sum(cluster_sizes[cluster_idx + 1:]))])
        avg_silhouettes[cluster_idx] = torch.sum(silhouettes*mask)/cluster_sizes[cluster_idx]

    return avg_silhouettes

# https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
def db_index(centroids, clusters):
    print("Calculating Davies-Bouldin Index")

    # calculate pairwise distances between cluster centroids
    m_ij = torch.cdist(centroids, centroids)

    avg_distances = torch.zeros(len(centroids))
    for cluster_idx in range(len(centroids)):
        # calculate average distance between a point and it's respective centroid
        avg_distances[cluster_idx] = torch.mean(torch.cdist(clusters[cluster_idx], centroids[cluster_idx, None].to(torch.float)))
    
    # compute pairwise sums
    pairwise_sum_avg_dist = torch.cdist(avg_distances[:, None], -1*avg_distances[:, None], p = 1)
    del avg_distances

    # compute R_i,j values
    r_ij = (pairwise_sum_avg_dist/m_ij).fill_diagonal_(0)
    del m_ij
    del pairwise_sum_avg_dist

    # compute D_i values by taking columnwise maximum
    d_i = torch.max(r_ij, dim=0).values
    del r_ij

    # compute Davies Bouldin index by taking the average of d_i values
    db_idx = torch.mean(d_i)

    return db_idx

    
    
