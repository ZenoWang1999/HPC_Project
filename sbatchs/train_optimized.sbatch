#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=64GB
#SBATCH --job-name=optimized
#SBATCH --output=optimized.out
module purge
module load python/intel/3.8.6
module load cuda/11.6.2
module load cudnn/8.6.0.163-cuda11
source $SCRATCH/myProject/bin/activate
python ../optimized_model.py