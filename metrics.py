import torch.distributed as dist
import os
import math
import torch
from tqdm import tqdm

def inertia(centroids, clusters):
    inertias = torch.zeros(len(centroids)).to(device="cuda")

    # calculate the inertias for nodes which belong to this rank
    print("calculating inertias")
    for idx in tqdm(range(dist.get_rank(), len(centroids), dist.get_world_size())):
        centroid = torch.unsqueeze(centroids[idx], 0).to(device="cuda")
        cluster = clusters[idx].to(device="cuda").double()
        inertia = torch.mean(torch.square(torch.cdist(centroid, cluster)))

        inertias[idx] = inertia

        # gather up the tensors
        dist.gather(inertias[idx], [inertias[a] for a in range(idx, idx+dist.get_world_size())])

        # clean up
        del centroid
        del cluster
        del inertia
    
    inertias.to(device="cpu")
    return inertias

# see https://en.wikipedia.org/wiki/Silhouette_(clustering) for a/b notation
def silhouette_coef(centroids, clusters):
    mean_si_over_clusters = torch.zeros(len(centroids))
    for cluster_idx in range(len(centroids)):
        # information about the current cluster
        curr_cluster = clusters[cluster_idx]

        # initialize tensors to hold a and b values
        a_values = torch.zeros(len(curr_cluster))
        b_values = torch.zeros(len(curr_cluster))

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
        

    silhouette_coef = max(mean_si_over_clusters)
    return silhouette_coef



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
        if (len(curr_cluster) > 1000):
            # if the GPU can't handle a big cluster, process it in blocks
            block_size = 1000
            running_b = torch.tensor([]).to(device="cuda")
            for idx in range(0, len(curr_cluster) + 1, block_size):
                bottom_of_range = idx
                top_of_range = idx + block_size
                if cluster_idx < top_of_range and cluster_idx > bottom_of_range:
                    other_centroids = torch.cat((centroids[bottom_of_range:cluster_idx], centroids[cluster_idx + 1:min(top_of_range, len(centroids[cluster_idx]))])).to(device="cuda")
                else:
                    other_centroids = centroids[bottom_of_range:top_of_range].to(device="cuda")
                b = torch.min(torch.cdist(curr_cluster, other_centroids), dim=1, keepdim=True).values
                del other_centroids
                running_b = torch.min(torch.cat((running_b, b), dim=1), dim=1, keepdim=True).values
                del b
            b = running_b
        else:
            other_centroids = torch.cat((centroids[:cluster_idx], centroids[cluster_idx + 1:])).to(device="cuda")
            b = torch.min(torch.cdist(curr_cluster, other_centroids), dim=1, keepdim=True).values

        # clean up a lil bit
        del curr_cluster

        # calculate average s value for this cluster
        s_values = (b - a)/float(max(torch.cat((a, b))))
        curr_sill_val = sum(s_values)/len(centroids)
        if curr_sill_val > silhouette_coef:
            silhouette_coef = curr_sill_val
        
        # clean up again
        del a
        del b
        del curr_sill_val

    # gather up the max silhouette coef
    dist.reduce(silhouette_coef, 0, dist.ReduceOp.MAX)

    return silhouette_coef
    print(silhouette_coef)


    
    
