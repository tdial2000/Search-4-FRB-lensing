#!/bin/bash
#SBATCH --time=20:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=32

# directory arguments
config=$1


# run code 

srun /fred/oz002/tdial/frb_lensing/src/_run_search4frblensing.sh $config