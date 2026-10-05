#!/bin/bash

#SBATCH -J supervised_training_kfold
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_%a_kfold%j.out  # %a for array job index
#SBATCH -e script_logs/err_%a_kfold%j.err
#SBATCH --export=none

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=7G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1  # 1 GPU per job
#SBATCH --partition=gpu_a100
#SBATCH --qos=gpu_all
#SBATCH --time=15:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --array=0-4  # 5 folds (indices 0-4), but SLURM will run trainings in parallel based on available GPUs and user's job limit

mkdir -p script_logs

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi

echo "Launch python file for fold $SLURM_ARRAY_TASK_ID"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH=$(dirname $(which pixi)):$PATH

srun --export=ALL pixi run -e gpu python mains/main_supervised_training.py \
    --config_train configs/models/train_trex.yaml \
    --config_model configs/models/ALISE.yaml \
    --fold_index $SLURM_ARRAY_TASK_ID  # Pass the fold index from the array

echo "End"
