import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from argparse import ArgumentParser
import numpy as np
import os
import math
from metrics import inertia, simplified_silhouette, db_index, validation_simple_silhouettes
import pickle

def main(config, BASE_PATH, CLUSTER_SET, PATH_TO_STORED_METRICS):
    # initialize a process group
    world_size = int(os.environ["WORLD_SIZE"])
    # the 'rank' of the current process, a unique id for the current gpu
    rank = int(os.environ["RANK"])
    # print(rank)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


    # load the embeddings
    embeddings_path=cfg.embeddings_path 
    embeddings=np.load(embeddings_path, mmap_mode="r")

    for LEVEL in range(1, config.n_levels + 1):
        # clusters in the current level
        clusters = np.load(f'{BASE_PATH}/{CLUSTER_SET}/level{LEVEL}/sorted_clusters.npy', allow_pickle=True)

        # get the centroids for the current level
        centroids = torch.from_numpy(np.load(f'{BASE_PATH}/{CLUSTER_SET}/level{LEVEL}/centroids.npy', allow_pickle=True))

        # if the level isn't the first one, get the cluster indices from the previous level and flatten them
        curr_level_clusters = []

        # read the embeddings if it's the first level, or read in the previous level's clusters if it's not
        if LEVEL == 1:
            for c in range(cfg.n_clusters[LEVEL - 1]):
                this_cluster = torch.tensor(embeddings[clusters[c]])
                curr_level_clusters.append(this_cluster)
            
        else:
            for c in range(cfg.n_clusters[LEVEL - 1]):
                this_cluster = torch.empty((0,768))
                for prev_cluster_idx in clusters[c]:
                    this_cluster = torch.cat((this_cluster, previous_level_clusters[prev_cluster_idx]))
                curr_level_clusters.append(this_cluster)


        previous_level_clusters = curr_level_clusters

        # print(len(curr_level_clusters))

        # calculate the inertia of this level's clusters
        storage_path = f'{PATH_TO_STORED_METRICS}/{CLUSTER_SET}/level{LEVEL}/'

        inertia_tensor = inertia(centroids, curr_level_clusters)
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        with open(storage_path + 'inertia.pickle', 'wb') as file:
            pickle.dump(inertia_tensor, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f'Inertias calculated for level {LEVEL}')

        # calculate the simplified silhouette coefficients of this clustering
        silhouette_tensor = simple_silhouette_easier(centroids, curr_level_clusters)
        with open(storage_path +'silhouette_coefficients.pickle', 'wb') as file:
            pickle.dump(silhouette_tensor, file, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'Silhouettes calculated for level {LEVEL}')
        if LEVEL != 1:
            db_tensor = db_index(inertia_tensor, centroids)
            with open(storage_path +'db_index.pickle', 'wb') as file:
                pickle.dump(silhouette_tensor, file, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'Davies-Bouldin Index calculated for level {LEVEL}')

        # clean up
        del inertia_tensor
        




if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path to config file", default="./4_level_skyline_5_cat/config.yaml")
    parser.add_argument("--world_size", type=int, help="Path to config file", default="1")
    parser.add_argument("--cluster_set", type=str, help="Path to config file", default="4_level_skyline_5_cat")
    parser.add_argument("--path_to_stored_metrics", type=str, help="Path to config file", default="/home/luke/Documents/metrics/outputs")
    parser.add_argument("--base_path", type=str, help="Path to config file", default="/home/luke/Documents/metrics")
    parser.add_argument("--embeddings_path", type=str, help="Path to config file", default="./skyline_embeddings_64.npy")

    args, opts = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(vars(args)),
        OmegaConf.from_cli(opts),
    )

    main(cfg, args.base_path, args.cluster_set, args.path_to_stored_metrics)
    dist.destroy_process_group()
