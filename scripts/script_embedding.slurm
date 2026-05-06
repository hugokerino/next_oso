#!/bin/bash

#SBATCH -J alise_embedding
#SBATCH --array=1-5:1%5
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 02:00:00
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_emb%j.out
#SBATCH -e script_logs/err_emb%j.err
#SBATCH --export=none

mkdir -p script_logs

PATH_s2='/work/THEIA/CES/oso/hugok/s2_data/'
PATH_mnt='/work/THEIA/CES/oso/hugok/mnt_data'
PATH_label='/work/THEIA/CES/oso/hugok/labels'
PATH_class_name='/work/THEIA/CES/oso/hugok/files_s2/oso_classes.csv'
PATH_logs='/work/THEIA/CES/oso/hugok/next_oso/logs/rf_training'
PATH_save_embedding='/work/THEIA/CES/oso/hugok/alise_weights/embeddings'

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi
#module load cuda/11.7

echo "Launch python file"

pixi run python Alise/generate_embedding.py \
    --path_s2 "$PATH_s2" \
    --path_mnt "$PATH_mnt" \
    --path_label "$PATH_label" \
    --path_oso_classes "$PATH_class_name" \
    --path_logs "$PATH_logs" \
    --path_saved_embeddings "$PATH_save_embedding"
    --fold ${SLURM_ARRAY_TASK_ID} \
echo "End"
