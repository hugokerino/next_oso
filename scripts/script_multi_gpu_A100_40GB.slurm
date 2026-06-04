#!/bin/bash

#SBATCH -J main_A100
#SBATCH -A cesbio
#SBATCH -o script_logs/out_main_a100_%j.out
#SBATCH -e script_logs/err_main_a100_%j.err
#SBATCH --export=none

#SBATCH -N 1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000M 

#SBATCH --gpus=a100_40g:2 
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
#module load cuda/11.7

MODELNAME="JOB_monitoring"

module load monitoring/3.0
start_monitoring.sh --name $MODELNAME --io local

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH=$(dirname $(which pixi)):$PATH

echo "Launch python file"

srun --export=ALL --ntasks-per-node=2 --cpus-per-task=16 pixi run -e gpu python mains/main_lightning.py \
    --config_train configs/train_trex.yaml \
    --config_model configs/models/UTAE.yaml
 
stop_monitoring.sh --name $MODELNAME

echo "End"
