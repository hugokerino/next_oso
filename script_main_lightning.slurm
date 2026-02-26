#!/bin/bash

#SBATCH -J main_lightning_ltae
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_main_light%j.out
#SBATCH -e script_logs/err_main_light%j.err
#SBATCH --export=none

#SBATCH -N 1                        # number of nodes ( or --nodes=)
#SBATCH -n 5                        # number of tasks ( or --tasks=)
#SBATCH --gpus=v100_32g:1           # archi and number of gpus 
#SBATCH --qos=gpu_all               # QoS
#SBATCH --partition=gpu_std         # partition
#SBATCH --time=10:00:00             # Walltime
#SBATCH --mem-per-cpu=12000M
  
mkdir -p script_logs

PATH_s2='/work/THEIA/CES/oso/hugok/s2_data/'
PATH_label='/work/THEIA/CES/oso/hugok/labels'
PATH_class_name='/work/THEIA/CES/oso/hugok/oso_classes.csv'
PATH_logs='/home/il/kerinoh/Documents/next_oso/logs/light_training'

echo "PATH_s2: $PATH_s2"
echo "PATH_label: $PATH_label"
echo "PATH_class_name: $PATH_class_name"
echo "PATH_logs: $PATH_logs"

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi

echo "Launch python file"

pixi run python main_lightning.py \
    --batch_size 8 \
    --epochs_max 5 \
    --path_s2 "$PATH_s2" \
    --path_label "$PATH_label" \
    --path_oso_classes "$PATH_class_name" \
    --path_logs "$PATH_logs"

echo "End"
