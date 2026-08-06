#!/bin/bash

#SBATCH -J pastis
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_pastis%j.out
#SBATCH -e script_logs/err_pastis%j.err
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

pixi run -e gpu python pastis/main_pastis.py \
    --config_train pastis/config_train.yaml \
    --config_model configs/models/ALISE.yaml

echo "End"
