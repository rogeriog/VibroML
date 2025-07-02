#!/bin/bash
#SBATCH --job-name=vibroml_phonons
#SBATCH --time=10:00:00
#SBATCH --output=log_phonons.txt
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
module load CUDA cuDNN/8.0.4.30-CUDA-11.1.1 
# TensorFlow/2.5.0-fosscuda-2020b
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/home/ucl/modl/rgouvea/anaconda3/envs/env_tfmodnet/lib/"
CUDA_DIR=/home/ucl/modl/rgouvea/anaconda3/envs/env_tfmodnet/lib/python3.8/site-packages/nvidia/cuda_nvcc/


conda activate /auto/globalscratch/users/r/g/rgouvea/vibroml_env
#export PYTHONUSERBASE=intentionally-disabled  ##it was loading local modnet...
echo "start"
date

vibroml --cif Cs2KInI6.cif --auto --soft_mode_num_top_structures_to_analyze 10
echo "--- Calculation Done ---"
date