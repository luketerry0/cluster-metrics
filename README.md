# Introduction

Throughout the past year, I have been working to provide more clarity and guidelines to the specific technique used in [this paper](https://arxiv.org/abs/2405.15613). This README will go into detail both about how I run the code which is included in that paper on OSCER (altered code can be found [here](https://github.com/luketerry0/ssl-data-curation)). I will also elaborate on the code contained in this repository, which calculates certain cluster metrics at each level of the hierarchy.

# Running Self-Supervised Data Curation on OSCER

This section details how I ran the ssl-data-curation code ([which can be found in this repo](https://github.com/luketerry0/ssl-data-curation)) on OSCER.

The original code in the repository is very well built, but it did not use WandB to to monitor the runs, nor did it initially run with our actual installation of SLURM. By altering code in `hierarchical_kmeans_launcher.py`, `run_disributed_kmeans.py`, I made it do both. This primarily entailed many small changes to the slurm scripts, but fundamentally changed nothing of substance. 

`spawn_new_hierarchy.sh` (in src/scripts) is a bash script which will generate all code to create a run a new clustering hierarchy. First, create a config file in the configs folder, and then call `spawn_new_hierarchy.sh` to create a new clustering and begin running it on OSCER. (Also, the bash script can be examined to see how a new clustering could be created manually). `hierarchical_kmeans_launcher.py` generates all the slurm scripts and the folder structure necessary to run the project, and then the slurm scripts use `run_distributed_kmeans.py` to initialize the clustering and log to wandb. Many areas of this project rely on the folder structure which is produced by a clustering, so use caution when changing that.

Also, you may need to change the mamba environments which the scripts refer to in `hierarchical_kmeans_launcher.py` and `spawn_new_hierarchy.sh` when I graduate. I have no idea what will happen to my OSCER home directory where the environment is located.

# Cluster Metrics Code

This repository currently contains code which will calculate the inertia of each cluster (exploiting the folder structure that the `ssl-data-curation` code produces). 

## How it works

This code:
- Reads the clustering
- Calculates pairwise distances between each point and each centroid (or, for performance, between each point and it's respective centroid)
- Stores those distances in a memory map
- Reads the distances and calculates the inertia
- Logs all of this, and a sample of images from the best/worst clusters according to inertia to WandB

Currently, the code will use a 1 dimensional memmap to store distances and only calculate inertia (for performance). 

Previously the code calculated all pair-wise distances between centroids and points. This would contain all the distances needed to calculate more sophisticated metrics like davies-bouldin index, silhouette-score, etc. [Functional code for this can be found in this commit; reverting to this commit will bring back the 2d distance calculation.](https://github.com/luketerry0/cluster-metrics/commit/c037cfea80d6430dd0be24054acaf38fa88785a1)

### Cluster Indexing Scheme
Once the other repository calculates the hierarchical code, `sorted_clusters.npy` contains a two dimensional list which relates the clusters at level one to the embeddings they contain (i. e. a value of 3 in the 0th row indicates that the embedding which was at index three of the originally passed .npy file is in the 0th cluster. 0th cluster refers to the index in `centroids.npy`). In subsequent levels, clusters are a combination of clusters in previous levels. So an index of 4 contained in cluster 3 at level 2 denotes that all of the embeddings contained in the 4th cluster at level 1 belong to the 3rd cluster in level 2. The level is not zero indexed, but the clusters and centroids are.

This indexing scheme is confusing, so if you intend to edit this code at all, I recommend closely reading the code in lines 40-65 of `main.py`, which manages this indexing. 

### Using the Memmap
Using the clustering, the code uses pytorch to calculate a small set of distances and store it in a [numpy memmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html). Using a memmap allows different processes to edit the memmap in parallel without any issues, so long as they don't try to change elements at the same index. A memmap essentially allows us to maintain a numpy array that is mapped directly to storage rather than memory. As a process changes a value in a memmap, it only stores in memory the values which it changes. When the `flush()` method is called, it writes these values to storage. This means we can have access to a numpy array that is larger than we could store in memory.

The drawback of using a memmap is that read/write opertaions take far longer when writing to storage rather than memory. The specific I/O times are highly sensitive to the size of the data being written as well as where the memmap is stored. LSCRATCH is the fastest on OSCER, followed by SCRATCH and OURDISK (but for the size of our data, LSCRATCH is the only real choice). 

Problems that will fail silently and take too long to find (BEWARE):

- If you maintain a two dimensional memmap, the default is row-major ordering. If you write multiple rows to this memmap, numpy will process each row as a seperate I/O transaction, destroying the efficiency of the code, and generally making you sad. To deal with this, the blocking scheme seen in `compute_distances()` (in `metrics.py`) lets you specify a block size. It will write rectangular blocks that contain as few rows as possible, and more importantly only sequential elements, resulting in one I/O transaction per read-write

- LSCRATCH is a physical SSD on each node (whereas OURDISK/SCRATCH are distributed filesystems). This means that a memmap on LSCRATCH will be different between processes on different nodes. This is crucial to consider when blocking and calculating metrics. Ensure that each distance is only calculated once, and use (distributed communication)[https://docs.pytorch.org/docs/stable/distributed.html] to calculate metrics across nodes. 

- Memmaps do not enforce shapes like regular numpy arrays! If you read in the file with different dimensions and parameters, the code will silently fail to do what you intended it to do.

To help you deal with new distributed code, the file `sanity_check.py` and `run_sanity_check_local.sh` can be used to run all the code here locally, on 1 GPU, and compare the output to scikit learn (with random data that is small enough to manually check). This can help with distribution, but you will still need to consider problems that may arise between nodes (and not just processes).

## Expanding the code

To expand the code, observe how the inertia method is used in `metrics.py` as well as `main`. Copy the same structure for the new metric, and change main to do whatever you want with it (like logging to wandb). 

## How to run it

After creating a clustering, just run `sbatch metrics.slurm CLUSTER_NAME` (where cluster name corresponds to the name of the config file of your clustering without '.yaml').

This will search the directory `/ourdisk/hpc/ai2es/luketerry/` for the clustering. If you've changed the `ssl-data-curation` code to put the clusters somewhere else, you will have to change the slurm script to refer to the new path.

Also, it attempts to read my WandB api key from a file which you can't access. In my home directory, I have a file called `.wandb_api_key.sh` which just contains `WANDB_API_KEY="my key here"` (with permissions locked down for security). Make your own file and change line 45 in the slurm script to refer to your file. If you like living a little on the edge, delete line 45 and replace the variable with your wandb api key in plain text, right out in the open where all the hackers, marauders, pirates, malefactors, and desperados on SCHOONER can see it. 

To run locally, use `local_run.sh` or `run_sanity_check_local.sh`. Both of these are useful for prototyping without dealing with queue times or debugging on OSCER. 

# Questions

If you have any questions about the code, feel free to email me at luke.terry0682@gmail.com or open an issue on the metrics github repository. I'll do my best to get back to you, but keep in mind that my best might not be very good. I hope the code is somewhat comprehensible on it's own. 


