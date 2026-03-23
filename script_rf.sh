#!/bin/bash

#SBATCH -J random_forest
#SBATCH --array=1-9:1%10
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 02:00:00
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_rf%j.out
#SBATCH -e script_logs/err_rf%j.err
#SBATCH --export=none

mkdir -p script_logs

PATH_s2='/work/THEIA/CES/oso/hugok/s2_data/'
PATH_label='/work/THEIA/CES/oso/hugok/labels'
PATH_class_name='/work/THEIA/CES/oso/hugok/files_s2/oso_classes.csv'
PATH_logs='/work/THEIA/CES/oso/hugok/next_oso/logs/rf_training'

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

zone_eco=${SLURM_ARRAY_TASK_ID}

pixi run python RandomForest/main_RF.py \
    --path_s2 "$PATH_s2" \
    --path_label "$PATH_label" \
    --path_oso_classes "$PATH_class_name" \
    --path_logs "$PATH_logs" \
    --zone_eco ${zone_eco} \
    --folds_train "[1, 2, 3]" \
    --folds_test "[5]" \

echo "End"
