#!/bin/bash

#SBATCH -J main_torch_ltae
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 20:00:00
#SBATCH -A theia_oso
#SBATCH -o logs/out_main_%j.out
#SBATCH -e logs/err_main_%j.err
#SBATCH --export=none

mkdir -p logs
mkdir -p res

PATH_s2='/work/THEIA/CES/oso/hugok/s2_data/'
PATH_label= '/work/THEIA/CES/oso/hugok/labels'
PATH_class_name= '/work/THEIA/CES/oso/hugok/oso_classes.csv'
PATH_logs= '/home/il/kerinoh/Documents/next_oso/logs/torch_training'

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi

echo "Launch python file"

pixi run python main_pytorch.py  --batch_size 8 --epochs_max 5 --path_s2 $PATH_s2 --path_label $PATH_label --path_oso_classes $PATH_class_name --path_logs $PATH_logs

echo "End"
