#!/bin/bash

#SBATCH -J main_lightning
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_main%j.out
#SBATCH -e script_logs/err_main%j.err
#SBATCH --export=none

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=7G

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --qos=gpu_all

#SBATCH --time=20:00:00             # Walltime

#SBATCH --signal=SIGUSR1@90         # For auto-resubmit

mkdir -p script_logs

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi
#module load cuda/11.7

echo "Launch python file"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH=$(dirname $(which pixi)):$PATH

srun --export=ALL pixi run -e gpu python mains/main_lightning.py \
    --config_train configs/train_trex.yaml \
    --config_model configs/models/LTAE.yaml
 
echo "End"
