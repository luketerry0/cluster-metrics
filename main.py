import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from argparse import ArgumentParser
import numpy as np
import os
import math
from inertia import inertia

def main(config):
    # initialize a process group
    WORLD_SIZE=1
    dist.init_process_group(backend="nccl", init_method="tcp://localhost:29500", rank=0, world_size=WORLD_SIZE)

    BASE_PATH = "/home/luke/Documents/metrics"
    CLUSTER_SET="4_level_skyline_5_cat"

    # load the embeddings
    embeddings_path=f'/home/luke/Documents/metrics/skyline_embeddings_64.npy'
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
                this_cluster = torch.tensor((embeddings[clusters[c]]))
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
        inertia(centroids, curr_level_clusters)


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path to config file", default="./4_level_skyline_5_cat/config.yaml")
    args, opts = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(vars(args)),
        OmegaConf.from_cli(opts),
    )

    main(cfg)
    dist.destroy_process_group()
