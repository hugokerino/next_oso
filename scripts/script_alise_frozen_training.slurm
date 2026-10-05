#!/bin/bash

#SBATCH -J alise_decoding
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_alise_dec%j.out
#SBATCH -e script_logs/err_alise_dec%j.err
#SBATCH --export=none

#SBATCH -N 1
#SBATCH --tasks=9
#SBATCH --mem-per-cpu=10G

#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_std
#SBATCH --qos=gpu_all

#SBATCH --time=15:00:00             # Walltime

#SBATCH --signal=SIGUSR1@90         # For auto-resubmit

mkdir -p script_logs

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi

echo "Launch python file"

pixi run -e gpu python mains/main_decoding.py \
    --path_embedding "/work/THEIA/CES/oso/hugok/alise_weights/embeddings" \
    --path_label "/work/THEIA/CES/oso/hugok/labels" \
    --path_mapping_folder "/work/THEIA/CES/oso/hugok/s2_data" \
    --path_logs "/work/THEIA/CES/oso/hugok/next_oso/logs/alise_decoding" \
    --batch_size 8

echo "End"
