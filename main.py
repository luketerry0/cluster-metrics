import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from argparse import ArgumentParser
import numpy as np
import os
import math
import wandb
from metrics import log_cluster, MetricsCalculator
import pickle
from time import perf_counter
# from torch.distributed.elastic.multiprocessing.errors import record

# @record
def main(config, BASE_PATH, CLUSTER_SET, PATH_TO_STORED_METRICS, KEY_PATH, FILEPATH_ORIGIN):
    # initialize a process group
    world_size = int(os.environ["WORLD_SIZE"])
    # the 'rank' of the current process, a unique id for the current gpu
    rank = int(os.environ["RANK"])
    d_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(d_id)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id = torch.device(f"cuda:{d_id}"))
    
    if dist.get_rank() == 0:
        print(f"FILEPATH_ORIGIN: {FILEPATH_ORIGIN}")
        print(f"KEY PATH: {KEY_PATH}")

        # initialize wandb logging
        run = wandb.init(
            project="ssl-clustering-metrics",
            entity='ai2es',
            name=args.wandb_name,
            # dir=f'/ourdisk/hpc/ai2es/luketerry/wandbruns/{args.wandb_name}/metrics/wandb',
            config=OmegaConf.to_container(cfg)
        )   


    # load the embeddings
    embeddings_path=cfg.embeddings_path 
    embeddings=np.load(embeddings_path, mmap_mode="r")
    embeddings_dim = embeddings[0].shape[0]

    # initialize wandb step
    current_step = 0

    for LEVEL in range(1, config.n_levels + 1): #TODO make this start at 1 again
        print(f"beginning level {LEVEL}")
        # clusters in the current level
        clusters = np.load(f'{BASE_PATH}/{CLUSTER_SET}/level{LEVEL}/sorted_clusters.npy', allow_pickle=True)

        # get the centroids for the current level
        centroids = torch.from_numpy(np.load(f'{BASE_PATH}/{CLUSTER_SET}/level{LEVEL}/centroids.npy', allow_pickle=True))

        # if the level isn't the first one, get the cluster indices from the previous level and flatten them
        curr_level_clusters = []
        cluster_indices = []

        # read the embeddings if it's the first level, or read in the previous level's clusters if it's not
        t1 = perf_counter()
        if LEVEL == 1:
            for c in range(cfg.n_clusters[LEVEL - 1]):
                this_cluster = torch.tensor(embeddings[clusters[c]])
                curr_level_clusters.append(this_cluster)
                cluster_indices.append(clusters[c])

        else:
            for c in range(cfg.n_clusters[LEVEL - 1]):
                curr_prev_clusters = []
                curr_prev_indices = []
                
                for prev_cluster_idx in clusters[c]:
                    curr_prev_clusters.append(previous_level_clusters[prev_cluster_idx])
                    curr_prev_indices.append(previous_cluster_indices[prev_cluster_idx])
                this_cluster = torch.cat(curr_prev_clusters)
                curr_level_clusters.append(this_cluster)
                cluster_indices.append(np.concatenate(curr_prev_indices))

        t2 = perf_counter()
        print(f"Time spent compiling embedding locations: {t2 - t1}")

        previous_level_clusters = curr_level_clusters
        previous_cluster_indices = cluster_indices

        # calculate the inertia of this level's clusters
        storage_path = f'{PATH_TO_STORED_METRICS}/{CLUSTER_SET}/level{LEVEL}/'

        metrics_calulator = MetricsCalculator(centroids, curr_level_clusters, embeddings_dim, FILEPATH_ORIGIN, config['block_size'])

        inertia_tensor = metrics_calulator.inertia()

        if dist.get_rank() == 0:
            # inertia_tensor = inertia(centroids, curr_level_clusters)
            t1 = perf_counter()
            if not os.path.exists(storage_path):
                os.makedirs(storage_path)
            t2 = perf_counter()
            with open(storage_path + 'inertia.pickle', 'wb') as file:
                pickle.dump(inertia_tensor, file, protocol=pickle.HIGHEST_PROTOCOL)
            t3 = perf_counter()

            print(f"Time spent making storage path {t2-t1}")
            print(f"Time spent storing inertia pickle: {t3-t2}")
            
            print(f'Inertias calculated for level {LEVEL}')

        # # calculate the simplified silhouette coefficients of this clustering
        # silhouette_tensor = simplified_silhouette(centroids, curr_level_clusters)
        # with open(storage_path +'silhouette_coefficients.pickle', 'wb') as file:
        #     pickle.dump(silhouette_tensor, file, protocol=pickle.HIGHEST_PROTOCOL)

        # print(f'Silhouettes calculated for level {LEVEL}')

        # db_tensor = db_index(centroids, curr_level_clusters)
        # with open(storage_path +'db_index.pickle', 'wb') as file:
        #     pickle.dump(db_tensor, file, protocol=pickle.HIGHEST_PROTOCOL)

        # print(f'Davies-Bouldin Index calculated for level {LEVEL}')

        # sample images from cluster with the highest and lowest inertias to wandb
        
        if dist.get_rank() == 0:
            # read in filepaths to pictures
            ta = perf_counter()
            with open(KEY_PATH, "rb") as fp:
                filepaths = pickle.load(fp)
            tb = perf_counter()
            print(f"Time spent loading filepaths pickle: {tb-ta}")

            # credit to https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement
            ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

            # log the best and worst 5 clusters based on inertia
            cluster_ranking = torch.argsort(inertia_tensor)
            worst_clusters = []
            sampling_threshold = 10
            n_sampled = 0
            curr_idx = 0
            # sample 5 best clusters
            while n_sampled < 5:
                cluster_idx = cluster_ranking[curr_idx]
                print(cluster_idx)
                print(inertia_tensor[cluster_idx])
                print(inertia_tensor)
                if (len(curr_level_clusters[cluster_idx]) >= sampling_threshold):
                    log_cluster(
                        filepaths, 
                        cluster_indices, 
                        cluster_idx,
                        f"{ordinal(n_sampled + 1)} best cluster, level {LEVEL}: inertia={inertia_tensor[cluster_idx]} (cluster idx: {cluster_idx})",
                        sampling_threshold,
                        LEVEL - 1
                    )
                    n_sampled += 1
                curr_idx += 1
                if curr_idx >= len(inertia_tensor):
                    print(f"No other clusters are big enough to be logged! Logged {curr_idx} clusters")
                    break
                print(f"time to log a best cluster: {tb - ta}")

            # sample 5 worst clusters
            curr_idx = -1
            n_sampled = 0
            while n_sampled < 5:
                cluster_idx = cluster_ranking[curr_idx]
                if (len(curr_level_clusters[cluster_idx]) >= sampling_threshold):
                    log_cluster(
                        filepaths, 
                        cluster_indices, 
                        cluster_idx,
                        f"{ordinal(n_sampled + 1)} worst cluster, level {LEVEL}: inertia={inertia_tensor[cluster_idx]}",
                        sampling_threshold,
                        LEVEL - 1
                    )
                    n_sampled += 1
                curr_idx -= 1
                if curr_idx*-1 >= len(inertia_tensor):
                    print(f"No other clusters are big enough to be logged! Logged {curr_idx} clusters")
                    break



            # log other things...
            inertia_list = inertia_tensor.tolist()
            wandb.log({"average inertia": sum(inertia_list)/len(inertia_list)}, step=LEVEL-1)
        #     print(f"Time to log avg. Inertia to wandb: {tc - tb}")
        #     del inertia_tensor
        # #     wandb.log({"davies bouldin index ": db_tensor.tolist()}, step = LEVEL-1)
        #     wandb.log({"simplified silhouette coeffecient": max(silhouette_tensor.tolist())}, step = LEVEL-1)

        # del db_tensor
        # del silhouette_tensor
        


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path to config file", default="./4_level_skyline_5_cat/config.yaml")
    parser.add_argument("--world_size", type=int, help="Path to config file", default="1")
    parser.add_argument("--cluster_set", type=str, help="Path to config file", default="4_level_skyline_5_cat")
    parser.add_argument("--path_to_stored_metrics", type=str, help="Path to config file", default="/home/luke/Documents/metrics/outputs")
    parser.add_argument("--base_path", type=str, help="Path to config file", default="/home/luke/Documents/metrics")
    parser.add_argument("--embeddings_path", type=str, help="Path to config file", default="./skyline_embeddings_64.npy")
    parser.add_argument("--wandb_name", type=str, help="wandb run name", default="no name passed")
    parser.add_argument("--filename_key_path", type=str, help="path to file which contains filenames that embeddings correspond to", default="no name passed")
    parser.add_argument("--filepath_origin", type=str, help="path where the very large distance matrix should be stored", default="no name passed")
    parser.add_argument("--block_size", type=int, help="size of computation blocks to use", default=10000)


    args, opts = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(vars(args)),
        OmegaConf.from_cli(opts),
    )

    print(f"fp path: {args.filename_key_path}")
    main(cfg, args.base_path, args.cluster_set, args.path_to_stored_metrics, args.filename_key_path, args.filepath_origin)
    dist.barrier()
    dist.destroy_process_group()
