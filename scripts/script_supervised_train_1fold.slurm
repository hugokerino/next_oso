#!/bin/bash

#SBATCH -J main_supervised_training_1fold
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_train%j.out
#SBATCH -e script_logs/err_train%j.err
#SBATCH --export=none

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=7G

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
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

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH=$(dirname $(which pixi)):$PATH

FOLD_INDEX=1  # For single fold training, we can set the fold index to 0

srun --export=ALL pixi run -e gpu python mains/main_supervised_training.py \
    --config_train configs/supervised_models/config_trex.yaml \
    --config_model configs/supervised_models/UTAE.yaml \
    --fold_index $FOLD_INDEX  # Pass the fold to train on

echo "End"
