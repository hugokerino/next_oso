#!/bin/bash

#SBATCH -J testing
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_testing%j.out
#SBATCH -e script_logs/err_testing%j.err
#SBATCH --export=none

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=7G

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --qos=gpu_all

#SBATCH --time=01:00:00             # Walltime

mkdir -p script_logs

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi

echo "Launch python file"

pixi run -e gpu python mains/main_testing.py \
    --config_train configs/models/train_trex.yaml \
    --config_model configs/models/ALISE.yaml \
    --path_dir_log /work/THEIA/CES/oso/hugok/next_oso/logs/light_training/ALISE_2026-06-01
echo "End"
