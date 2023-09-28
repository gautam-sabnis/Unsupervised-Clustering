#!/bin/bash
#SBATCH -p compute -q batch
#SBATCH --job-name="cluster-embeddings"
#SBATCH -N 1 # number of nodes
#SBATCH -n 60 # number of cores
#SBATCH --mem 512G
#SBATCH -t 01:30:00
#SBATCH -o /projects/kumar-lab/sabnig/Projects/Drinking/logs/cluster-embeddings.out # STDOUT
#SBATCH -e /projects/kumar-lab/sabnig/Projects/Drinking/logs/cluster-embeddings.err # STDOUT

source /home/sabnig/anaconda3/bin/activate
conda activate Drinking
python cluster_embeddings.py



