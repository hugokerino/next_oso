#!/bin/bash

#SBATCH -J testing_kfold
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_testing_kfoldV100%j.out
#SBATCH -e script_logs/err_testing_kfoldV100%j.err
#SBATCH --export=none

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=14G

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_std
#SBATCH --qos=gpu_all
#SBATCH --time=01:00:00             # Walltime

mkdir -p script_logs

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi

echo "Launch python file"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH=$(dirname $(which pixi)):$PATH

FOLD_INDEX=0  # For single fold training, we can set the fold index to 0

srun --export=ALL pixi run -e gpu python mains/main_supervised_testing.py \
    --config_train configs/models/model_trex.yaml \
    --config_model configs/models/ALISE.yaml \
    --path_dir_checkpoints /work/THEIA/CES/oso/hugok/next_oso/logs/light_training/ALISE_2026-07-17_kfold \
    --fold_index $FOLD_INDEX

echo "End"
