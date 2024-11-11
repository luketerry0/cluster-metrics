from omegaconf import OmegaConf
from argparse import ArgumentParser
import numpy as np
import pickle

def main(config):
    # load the embeddings
    embeddings_path="/home/luke/Documents/metrics/skyline_embeddings_64.npy"
    embeddings = np.load(embeddings_path)

    # construct an object which denotes the structure of the hierarchy, with embedding indices
    hierarchy = []

    # get embedding values and cluster values
    CLUSTER_SET = '4_level_skyline_5_cat'
    BASE_PATH = "."
    for LEVEL in range(config.n_levels):
        NUM_CLUSTERS = config.n_clusters[LEVEL]
        LEVEL += 1
        # keep track of clusters in current level
        curr_level_clusters = []
        clusters = np.load(f'{BASE_PATH}/{CLUSTER_SET}/level{LEVEL}/sorted_clusters.npy', allow_pickle=True)
        for CLUSTER in range(NUM_CLUSTERS):

            # get the current cluster in the initial level: corresponds to centroids in previous levels unless LEVEL == 1
            cluster_indices = clusters[CLUSTER]

            # convert these indices to actual embeddings
            while LEVEL > 1:
                indices = np.array([])
                previous_level_clusters = np.load(f'{BASE_PATH}/{CLUSTER_SET}/level{LEVEL - 1}/sorted_clusters.npy', allow_pickle=True)
                for centroid_index in cluster_indices:
                    # grab the corresponding cluster in the previous level and flatten the indices
                    indices = np.concatenate((indices, previous_level_clusters[int(centroid_index)]))
                LEVEL -= 1
                cluster_indices = indices.astype(int)
            curr_level_clusters.append(cluster_indices)
        
        # get the centroids for the current cluster
        centroids = np.load(f'{BASE_PATH}/{CLUSTER_SET}/level{LEVEL}/centroids.npy')
        hierarchy.append({'centroids': centroids, 'clusters': curr_level_clusters, 'level': LEVEL})
        
    # calculate the statistics we want
    print(len(hierarchy[0]["clusters"]))
            



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