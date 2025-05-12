#!/bin/bash

#SBATCH --array=0-6
#SBATCH --time=5:00
#SBATCH --mem=12GB 
#SBATCH --cpus-per-task=32

# List of channelisations
# MAKE SURE TO CHANGE --array=0-(X-1) where X is the number of 
# channelisations or the length of [channel_list]!!
channel_list=(32 84 168 336 672 1024 1680)

# config filepath
config="/fred/oz002/tdial/frb_lensing/testing/sig0.15_dm0.1_del0.00001.txt"

srun bash -c "cd /fred/oz002/tdial/frb_lensing/src && source ~/vpython/bin/activate && python3 search4frblensing.py -c ${config} --nFFT ${channel_list[${SLURM_ARRAY_TASK_ID}]} -n sig0.15_dm0.1_del0.00001_nFFT${channel_list[${SLURM_ARRAY_TASK_ID}]}"


