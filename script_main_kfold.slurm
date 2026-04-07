#!/bin/bash

#SBATCH -J main_lightning_kfold
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_kfold%j_%a.out  # %a for array job index
#SBATCH -e script_logs/err_kfold%j_%a.err
#SBATCH --export=none

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=7G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1  # 1 GPU per job
#SBATCH --partition=gpu_a100
#SBATCH --qos=gpu_all
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --array=0-4  # 5 folds (indices 0-4), but SLURM will run up to 2 in parallel based on available GPUs

mkdir -p script_logs

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi

echo "Launch python file for fold $SLURM_ARRAY_TASK_ID"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH=$(dirname $(which pixi)):$PATH

srun --export=ALL pixi run -e gpu python main_kfold.py \
    --config_train configs/train_trex_kfold.yaml \
    --config_model configs/models/UTAE.yaml \
    --fold_index $SLURM_ARRAY_TASK_ID  # Pass the fold index from the array

echo "End"
