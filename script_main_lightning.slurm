#!/bin/bash

#SBATCH -J main_lightning_utae
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_main_utae%j.out
#SBATCH -e script_logs/err_main_utae%j.err
#SBATCH --export=none

#SBATCH -N 1
#SBATCH --tasks=9
#SBATCH --mem-per-cpu=10G

#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_std
#SBATCH --qos=gpu_all

#SBATCH --time=12:00:00             # Walltime

#SBATCH --signal=SIGUSR1@90         # For auto-resubmit

mkdir -p script_logs

PATH_s2='/work/THEIA/CES/oso/hugok/s2_data/'
PATH_label='/work/THEIA/CES/oso/hugok/labels'
PATH_class_name='/work/THEIA/CES/oso/hugok/files_s2/oso_classes.csv'
PATH_logs='/work/THEIA/CES/oso/hugok/next_oso/logs/light_training'

echo "PATH_s2: $PATH_s2"
echo "PATH_label: $PATH_label"
echo "PATH_class_name: $PATH_class_name"
echo "PATH_logs: $PATH_logs"

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi
#module load cuda/11.7

echo "Launch python file"

pixi run -e gpu python main_lightning.py \
    --model "UTAE" \
    --batch_size 4 \
    --epochs_max 50 \
    --accelerator "gpu" \
    --devices 1 \
    --nodes 1 \
    --strategy "ddp_find_unused_parameters_true" \
    --path_s2 "$PATH_s2" \
    --path_label "$PATH_label" \
    --path_oso_classes "$PATH_class_name" \
    --path_logs "$PATH_logs" \
    --label_type "probas"

echo "End"
