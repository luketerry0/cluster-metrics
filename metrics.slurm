#!/bin/bash
# the name of the partition you are submitting to
#SBATCH --partition=ai2es,normal
# the number of nodes you will use, usually only 1 is required.  If your code is not designed to use more then more will not be better.
#SBATCH --nodes=2
# the number of processes you will launch.  This should be equal to the number of nodes
#SBATCH --ntasks=2
# Thread count, or the number of hypercores you plan to use.  This is not enforced.
#SBATCH --cpus-per-task=64
# The number of gpus you require each node to have
#SBATCH --gres=gpu:2
# memory (RAM) you will use on each machine.  Provide an upper bound, this is not enforced
#SBATCH --mem=64G
# Where you would like your stdout and stderr to appear
#SBATCH --output=/home/luketerry/cluster-metrics/logs/out-%j.txt
#SBATCH --error=/home/luketerry/cluster-metrics/logs/err-%j.txt
# The maximum time your job can take (most partitions limit this)
#SBATCH --time=12:00:00
# job name which will appear in queue
#SBATCH --job-name=metrics
# if you fill this out slurm will email you job status updates, consider sending them to a folder.
#SBATCH --mail-user=luke.h.terry-1@ou.edu
#SBATCH --mail-type=ALL
# the working directory for your job.  This must exist.
#SBATCH --chdir=/home/luketerry/cluster-metrics/
#SBATCH --exclude=c829,c830
#################################################

# this is how we get the ip address of the node that will be used for the master process
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b")

echo Node IP: $head_node_ip

# using Dr. Fagg's conda setup script
. /home/fagg/tf_setup.sh
# activating a version of my environment
conda activate /home/luketerry/miniforge3/envs/ssl-data-curation

trial="$1" # something like "4_level_skyline_exp"

# logging in to weights and biases
source /home/luketerry/.wandb_api_key.sh
wandb login $WANDB_API_KEY

# copy our data to LSCRATCH to run
start_time=$(date +%s.%N)
sbcast /ourdisk/hpc/ai2es/luketerry/npy_laion_embeddings/laion_embeddings_shard_0.npy /lscratch/$SLURM_JOBID/embeddings.npy
chmod 777 /lscratch/$SLURM_JOBID/embeddings.npy
mid_time=$(date +%s.%N)
sbcast /ourdisk/hpc/ai2es/luketerry/npy_laion_embeddings/filenames_shard_0.pkl /lscratch/$SLURM_JOBID/filename-key.npy
chmod 777 /lscratch/$SLURM_JOBID/filename-key.npy
end_time=$(date +%s.%N)
t1=$(echo "scale=9; $mid_time - $start_time" | bc)
t2=$(echo "scale=9; $end_time - $mid_time" | bc)
ls /lscratch/$SLURM_JOBID/
echo "Time spent moving embeddings: $t1 seconds"
echo "Time spent moving filename key: $t2 seconds"

mkdir /scratch/luketerry/$SLURM_JOBID

# launching a run that will be executed over multiple compute nodes
echo $SLURM_GPUS_PER_NODE
srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint "$head_node_ip:64425" \
main.py \
--wandb_name="$1" \
--config_file="/ourdisk/hpc/ai2es/luketerry/${trial}/config.yaml" \
--world_size="$SLURM_JOB_NUM_NODES" \
--cluster_set="${trial}" \
--path_to_stored_metrics="/ourdisk/hpc/ai2es/luketerry/metrics" \
--base_path="/ourdisk/hpc/ai2es/luketerry" \
--embeddings_path="/lscratch/$SLURM_JOBID/embeddings.npy" \
--filename_key_path="/lscratch/$SLURM_JOBID/filename-key.npy" \
--filepath_origin="/scratch/luketerry/$SLURM_JOBID"
