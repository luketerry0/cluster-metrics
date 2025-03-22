import torch.distributed as dist
import torch.distributed.tensor
import os
import math
import torch
import numpy as np
from multiprocessing import shared_memory
import tempfile
from tqdm import tqdm
import itertools
import wandb
from PIL import Image
import gc
from time import perf_counter
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import sys

"""
Iterable which fetchesn  a single block of the points/centroids to store in the memmap
"""
class BlockFetchDataset(Dataset):
    def __init__(self, points, centroids):
        # store/initialize data
        self.points = points.float()
        self.centroids = centroids.float()

        # determine the dimension of each block and the dimension of the data
        self.n_points = len(points)
        self.n_centroids = len(centroids)
        self.block_size = 11_000
        self.n_points_per_block = min(self.block_size, self.n_points) # max number of points per block
        self.n_centroids_per_block = min(self.block_size, self.n_centroids) # max number of centroids per block
        self.points_dim = math.ceil(self.n_points / self.n_points_per_block)
        self.centroids_dim = math.ceil(self.n_centroids / self.n_centroids_per_block)
        self.n_blocks = self.points_dim*self.centroids_dim

        print(f"Initialized iterable, {self.n_points} points, {self.n_centroids} centroids, {self.n_blocks} blocks")


    def __iter__(self):
        return self

    def __getitem__(self, idx):
        # fetch the block
        points_lower_idx = (idx % self.points_dim)*self.n_points_per_block
        centroids_lower_idx = (idx // self.points_dim)*self.n_centroids_per_block
        
        points_upper_idx = min(points_lower_idx + self.n_points_per_block, self.n_points)
        centroids_upper_idx = min(centroids_lower_idx + self.n_centroids_per_block, self.n_centroids)

        block_points = self.points[points_lower_idx:points_upper_idx, ].to(device="cuda")
        block_centroids = self.centroids[centroids_lower_idx:centroids_upper_idx, ].to(device="cuda")

        # print(f"fetching: {points_lower_idx} to {points_upper_idx} points, {centroids_lower_idx} to {centroids_upper_idx} centroids")
        return block_points, block_centroids, points_upper_idx, points_lower_idx, centroids_upper_idx, centroids_lower_idx, idx

    def __len__(self):
        return self.n_blocks

"""
Class to calculate various metrics about a clustering
"""
class MetricsCalculator:
    def __init__(self, centroids, cluster_embeddings, embeddings_dim, file_prefix="./"):
        self.dtype = torch.float32
        self.centroids = centroids.to(self.dtype)
        self.file_prefix = file_prefix
        self.cluster_embeddings = cluster_embeddings
        self.embeddings_dim = embeddings_dim
        self.NP_SHARED_NAME = 'shared_distance_matrix'
        self.distances = self.compute_distances(centroids, cluster_embeddings, embeddings_dim)


    """
    clean up by removing the object containing pairwise distances
    """
    # def __del__(self):
    #     if dist.get_rank() == 0:
    #         del self.distances
    #         gc.collect()
    #         if os.path.exists(self.file_prefix + '/dists.memmap'):
    #             os.remove(self.file_prefix + '/dists.memmap')

    """
    helper function to write to a memmap in a new thread
    """
    def write_to_memmap(self, x_dim, y_dim, al, au, bl, bu, data, idx):
        dest = np.memmap(
            self.file_prefix + '/dists.memmap', 
            dtype='float32', mode='r+', 
            shape=(x_dim, y_dim),
            )
        dest[al:au, bl:bu] = data[:, :]
        dest.flush()
        del dest
        print(f"writing points {al} to {au} and cents {bl} to {bu}")
        return True

    """
    computes distances between all points and centroids, and stores their values in a memmaped numpy array for easy access
    """
    def compute_distances(self, centroids, clusters, embeddings_dim):
        # time this whole process
        t1 = perf_counter() 
        # flatten clusters into one long list of points
        points = torch.cat(clusters).to(self.dtype)
        # create a memory mapped array to store the result in
        t_mem_start = perf_counter()
        x_dim = len(points)
        y_dim = len(centroids)
        print(f"DIMENSIONS: {x_dim}, {y_dim}")
        if (dist.get_rank() % int(os.environ["LOCAL_WORLD_SIZE"])) == 0:
            distances_array = np.memmap(
                self.file_prefix + '/dists.memmap', 
                dtype='float32', mode='w+', 
                shape=(x_dim,y_dim))
            del distances_array
            print(f"rank {dist.get_rank()} created map at {self.file_prefix + '/dists.memmap'}")
        print(f"rank {dist.get_rank()} at checkpoint")
        dist.barrier()
        distances_array = np.memmap(
            self.file_prefix + '/dists.memmap', 
            dtype='float32', mode='r+', 
            shape=(x_dim,y_dim)) 
        t_mem_end = perf_counter()
        print(f"Time taken to create distances memmap: {t_mem_end - t_mem_start}")

        # use distributed sampler to fetch the data
        blockDataSet = BlockFetchDataset(points, centroids)
        blockSampler = DistributedSampler(blockDataSet, shuffle=False)
        blockLoader = DataLoader(blockDataSet, batch_size = 1, sampler=blockSampler)
        n_centroids = len(clusters)

        t_begin = perf_counter()
        coroutines = []
        with ThreadPoolExecutor(max_workers=64) as executor:
            for block_points, block_centroids, pu, pl, cu, cl, idx in blockLoader:
                # calculate distances for this block
                distances = torch.cdist(block_points, block_centroids).cpu()

                # store the distances in the correct area of the distances array
                coroutines.append(
                    executor.submit(self.write_to_memmap, x_dim, y_dim ,pl, pu,cl, cu,distances, idx)
                )

                # clean up
                del block_points
                del block_centroids
                del distances
                gc.collect()

        # wait for all the coroutines to finish
        test = concurrent.futures.wait(coroutines)
        torch.distributed.barrier()
        


        t_bflush = perf_counter()
        distances_array.flush()
        t_aflush = perf_counter()
        print(f"Time to flush memmap: {t_aflush - t_bflush}")

        # write from the copy on write memmap to the real one
        t_after = perf_counter()
        print(f"Time to iterate over all blocks on rank {dist.get_rank()}: {t_after - t_begin}")
        del distances_array
        distances_array = np.memmap(
            self.file_prefix + '/dists.memmap', 
            dtype='float32', mode='r', 
            shape=(len(points),len(centroids)))

        return distances_array

    """
    fetch the inertia, which is the row-wise minimum of the total distances array
    """
    def inertia(self):
        print("retrieving inertias")
        t1 = perf_counter()

        # use clusters to aggregate inertias
        cluster_sizes = [len(self.cluster_embeddings[i]) for i in tqdm(range(len(self.cluster_embeddings)))]
        cluster_indices = list(itertools.chain([0], itertools.accumulate(cluster_sizes)))
        cluster_inertias = []

        for i in tqdm(range(len(cluster_indices) - 1)):
            relevant_inertias = self.distances[cluster_indices[i]:cluster_indices[i+1], i]**2
            iner = np.sum(relevant_inertias)
            cluster_inertias.append(iner)


        t2 = perf_counter()
        iner_tensor = torch.tensor(cluster_inertias).to(device="cuda")
        # print(f"DIST: {dist.get_rank()}, inertia: {iner_tensor}")
        # ensure that the correct inertia is calculated, even if ranks don't have the whole memmap
        dist.all_reduce(iner_tensor, op=dist.ReduceOp.SUM)
        # corrects for the inertia being counted multiple times if more than one rank shares a node
        iner_tensor = iner_tensor / int(os.environ["LOCAL_WORLD_SIZE"])

        if dist.get_rank() == 0:
            print("INERTIA::")
            print(iner_tensor)
            print(cluster_sizes)
        dist.barrier()
        return iner_tensor

# """
# Calculates the inertia of this clustering

# centroids: a 2d tensor of the centroids
# clusters: a list of 2d tensors representing the contents of each cluster
# """
# def inertia(centroids, clusters):
#     inertias = torch.zeros(len(centroids)).to(device="cuda")

#     # calculate the inertias for nodes which belong to this rank
#     print("calculating inertias")
#     rank = dist.get_rank()
#     for idx in tqdm(range(rank, len(centroids), dist.get_world_size())):
#         centroid = torch.unsqueeze(centroids[idx], 0).to(device="cuda")
#         cluster = clusters[idx].to(device="cuda").double()
#         inertia = torch.sum(torch.square(torch.cdist(centroid, cluster)))

#         inertias[idx] = inertia

#         # clean up
#         del centroid
#         del cluster
#         del inertia

#     # gather up the tensors, and resize appropriately to calculate inertias
#     if rank == 0:
#         all_inertias = [torch.zeros(len(centroids)).to(device="cuda") for a in range(dist.get_world_size())]
#         dist.gather(inertias, all_inertias)
#         del inertias
#         stacked_inertias = torch.stack(all_inertias)
#         final_inertias = torch.sum(stacked_inertias, dim=0)
#         return final_inertias
#     else:
#         dist.gather(inertias)
#         del inertias



# # # see https://en.wikipedia.org/wiki/Silhouette_(clustering) for a/b notation
# def silhouette_coef(centroids, clusters):
#     mean_si_over_clusters = torch.zeros(len(centroids)).to(device="cuda")
#     for cluster_idx in range(len(centroids)):
#         # information about the current cluster
#         curr_cluster = clusters[cluster_idx]

#         # initialize tensors to hold a and b values
#         a_values = torch.zeros(len(curr_cluster)).to(device="cuda")
#         b_values = torch.zeros(len(curr_cluster)).to(device="cuda")

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
        

#     silhouette_coef = mean_si_over_clusters # = max(mean_si_over_clusters)
#     return silhouette_coef

# """
# Calculates the silhouette coeffiecient of the clustering

# centroids: a 2d tensor of the centroids
# clusters: a list of 2d tensors representing the contents of each cluster
# """
# def simplified_silhouette(centroids, clusters):

#     print("calculating silhouettes")
#     avg_silhouettes = torch.zeros(len(centroids))
#     for cluster_idx in tqdm(range(dist.get_rank(), len(centroids), dist.get_world_size())):
#         centroid = torch.unsqueeze(centroids[cluster_idx], 0).double().to(device="cuda")
#         curr_cluster = clusters[cluster_idx].to(device="cuda").double()

#         # calculate a' values
#         a = torch.cdist(curr_cluster, centroid).to(device="cuda")
#         del centroid

#         # calculate b' values
#         block_size = 3000
#         other_centroids = torch.cat((centroids[:cluster_idx], centroids[cluster_idx + 1:])).to(device="cuda")
#         if (len(curr_cluster) > block_size):
#             # if the GPU can't handle a big cluster, process it in blocks
#             running_pairwise_distances = torch.tensor([]).to(device="cpu")
#             for idx in range(0, len(curr_cluster) + 1, block_size):
#                 bottom_of_range = idx
#                 top_of_range = min(idx + block_size, len(curr_cluster))

#                 curr_cluster_block = curr_cluster[bottom_of_range:top_of_range]

#                 pairwise_distances = torch.cdist(curr_cluster_block, other_centroids).to(device="cpu")
#                 running_pairwise_distances = torch.cat((running_pairwise_distances, pairwise_distances))
#                 del curr_cluster_block
#             running_pairwise_distances.to(device="cuda")
#             b = torch.min(running_pairwise_distances, dim=1, keepdim=True).values.to(device="cuda")
#             del running_pairwise_distances
#         else:
#             pairwise_distances = torch.cdist(curr_cluster, other_centroids)
#             b = torch.min(pairwise_distances, dim=1, keepdim=True).values
#         # clean up a lil bit
#         del curr_cluster
#         del other_centroids

#         # calculate average s value for this cluster
#         s_values = torch.div((b - a).t(), torch.max(torch.cat((a, b), dim=1), dim=1).values)
#         avg_silhouettes[cluster_idx] = torch.mean(s_values)

#         # clean up again
#         del a
#         del b

#     return avg_silhouettes

# """
# Calculates the same simplified silhouette as above, but is easier to understand.
# Will crash because of cuda memory if ran on large data (this code is for validation only)

# centroids and clusters are same parameters as above
# """
# def validation_simple_silhouettes(centroids, clusters):
#     centroids = centroids
#     points = torch.cat(clusters).double()
#     cluster_sizes = [a.size(0) for a in clusters]

#     # calculate the pairwise distances between each point and every centroid, discarding all except the closest and second closest
#     # we're relying here on the fact that in k-means, the closest centroid always corresponds to the point's class
#     distances = torch.sort(torch.cdist(points, centroids), dim=1).values[:, :2]
#     del points
#     silhouettes = (distances[:,1] - distances[:,0]) / torch.max(distances, dim=1).values
#     # take the average of each cluster's silhouette 
#     cluster_sizes = [a.size(0) for a in clusters]
#     avg_silhouettes = torch.zeros(len(cluster_sizes))
#     for cluster_idx in range(len(cluster_sizes)):
#         mask = torch.cat(
#             [torch.zeros(sum(cluster_sizes[:cluster_idx])), 
#             torch.ones(cluster_sizes[cluster_idx]), 
#             torch.zeros(sum(cluster_sizes[cluster_idx + 1:]))])
#         avg_silhouettes[cluster_idx] = torch.sum(silhouettes*mask)/cluster_sizes[cluster_idx]

#     return avg_silhouettes

# # https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
# def db_index(centroids, clusters):
#     print("Calculating Davies-Bouldin Index")

#     # calculate pairwise distances between cluster centroids
#     m_ij = torch.cdist(centroids, centroids).to(device="cuda")

#     avg_distances = torch.zeros(len(centroids))
#     for cluster_idx in range(dist.get_rank(), len(centroids), dist.get_world_size()):
#         # calculate average distance between a point and it's respective centroid
#         avg_distances[cluster_idx] = torch.mean(torch.cdist(clusters[cluster_idx], centroids[cluster_idx, None].to(torch.float))).to(device="cuda")
#     del clusters

#     # compute pairwise sums
#     pairwise_sum_avg_dist = torch.cdist(avg_distances[:, None], -1*avg_distances[:, None], p = 1).to(device="cuda")
#     del avg_distances

#     # compute R_i,j values
#     r_ij = (pairwise_sum_avg_dist/m_ij).fill_diagonal_(0).to(device="cuda")
#     del m_ij
#     del pairwise_sum_avg_dist

#     # compute D_i values by taking columnwise maximum
#     d_i = torch.max(r_ij, dim=0).values.to(device="cuda")
#     del r_ij

#     # gather up the D_i values
#     dist.reduce(d_i, 0, torch.distributed.ReduceOp.SUM)

#     # compute Davies Bouldin index by taking the average of d_i values
#     if dist.get_rank() == 0:
#         db_idx = d_i[0]/len(centroids)
#         print(db_idx)
#         return db_idx

    
"""
Samples image from the cluster denoted by idx and logs them to wandb

filepaths: a list of tuples [(filepath, caption), ...] for the images
cluster_indices: list of tensors denoting the indices in the filepaths object where the images in a cluster are
   e. g. [(0, 1, 2, 3), (4, 5)] means the first four images in filepaths are in the first cluster, and the next two are in the second
idx: the index of the cluster to log
wandb_caption: the caption to log the image to wandb with
num_images: the number of images to log
log_step: the wandb step to log images with
"""
def log_cluster(filepaths, cluster_indices, idx, wandb_caption, num_images, log_step):
    # sample from the cluster and log to wandb
    curr_cluster_indices = cluster_indices[idx]
    # print(cluster_indices)
    sample = [Image.open(filepaths[curr_cluster_indices[x]][0]) for x in range(num_images)]

    # log the image(s) to wandb
    wandb.log({wandb_caption: [wandb.Image(img) for img in sample]}, step=log_step)

    
