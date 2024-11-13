from omegaconf import OmegaConf
from argparse import ArgumentParser
import numpy as np
import pickle
from inertia import inertia
import pandas as pd
from dask.distributed import Client
import dask.array as da
import dask
from dask_cuda import LocalCUDACluster

def main(config):
    # create dask client
    cluster=LocalCUDACluster()
    client=Client()

    BASE_PATH = "/home/luke/Documents/metrics"
    CLUSTER_SET="4_level_skyline_5_cat"

    # load the embeddings
    embeddings_path=f'/home/luke/Documents/metrics/skyline_embeddings_64.npy'
    embeddings = np.load(embeddings_path)

    # construct an object which denotes the structure of the hierarchy, with embedding indices
    hierarchy = []

    previous_level_clusters = np.load(f'{BASE_PATH}/{CLUSTER_SET}/level1/sorted_clusters.npy', allow_pickle=True)

    for LEVEL in range(1, config.n_levels + 1):
        # clusters in the current level
        clusters = np.load(f'{BASE_PATH}/{CLUSTER_SET}/level{LEVEL}/sorted_clusters.npy', allow_pickle=True)

        # get the centroids for the current level
        centroids = np.load(f'{BASE_PATH}/{CLUSTER_SET}/level{LEVEL}/centroids.npy', allow_pickle=True)

        # if the level isn't the first one, get the cluster indices from the previous level and flatten them
        curr_level_clusters = []

        if LEVEL != 1:
            for c in range(cfg.n_clusters[LEVEL - 1]):
                this_cluster = np.empty((0,768))
                for prev_cluster_idx in clusters[c]:
                    curr_prev_cluster = previous_level_clusters[prev_cluster_idx][1]
                    this_cluster = np.append(this_cluster, curr_prev_cluster, axis=0)
                curr_level_clusters.append([centroids[c], this_cluster])
        else:
            for c in range(cfg.n_clusters[LEVEL - 1]):
                this_cluster = np.array(embeddings[clusters[c]])
                curr_level_clusters.append([centroids[c], this_cluster]) #get_cluster(c, clusters[c], centroids, embeddings))
        previous_level_clusters = curr_level_clusters

        hierarchy.append({'clusters': curr_level_clusters, 'level': LEVEL})
        
    # calculate the statistics we want
    # print(inertia(hierarchy[1]['clusters'][0]))

    # result = client.submit(inertia, hierarchy[2]['clusters'][0])
    # print("!!!!!-----------------------------------------------------------------------------------------------------------------------------")
    # print(result.result())
    # print("!!!!!-----------------------------------------------------------------------------------------------------------------------------")

    a = client.map(inertia, hierarchy[3]['clusters'][:])
    print(client.gather(a))


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