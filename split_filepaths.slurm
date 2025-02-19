#!/bin/bash
# the name of the partition you are submitting to
#SBATCH --partition=ai2es,gpu
# the number of nodes you will use, usually only 1 is required.  If your code is not designed to use more then more will not be better.
#SBATCH --nodes=1
# the number of processes you will launch.  This should be equal to the number of nodes
#SBATCH --ntasks=1
# Thread count, or the number of hypercores you plan to use.  This is not enforced.
#SBATCH --cpus-per-task=32
# The number of gpus you require each node to have
#SBATCH --gres=gpu:0
# memory (RAM) you will use on each machine.  Provide an upper bound, this is not enforced
#SBATCH --mem=64G
# Where you would like your stdout and stderr to appear
#SBATCH --output=/home/luketerry/cluster-metrics/logs/out-fp-%j.txt
#SBATCH --error=/home/luketerry/cluster-metrics/logs/err-fp-%j.txt
# The maximum time your job can take (most partitions limit this)
#SBATCH --time=24:00:00
# job name which will appear in queue
#SBATCH --job-name=metrics_filepath_split
# if you fill this out slurm will email you job status updates, consider sending them to a folder.
#SBATCH --mail-user=luke.h.terry-1@ou.edu
#SBATCH --mail-type=ALL
# the working directory for your job.  This must exist.
#SBATCH --chdir=/home/luketerry/cluster-metrics/
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

python ./split_filepaths.py