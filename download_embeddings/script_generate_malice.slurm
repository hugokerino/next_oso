#!/bin/bash

#SBATCH -J malice_embedding
#SBATCH --array=1-5:1%5
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 10:00:00
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_emb%j.out
#SBATCH -e script_logs/err_emb%j.err
#SBATCH --export=none

mkdir -p script_logs

data='next_oso'
PATH_data='/work/THEIA/CES/oso/hugok/nextoso-dataset/data/10m' #'/work/THEIA/CES/oso/hugok/nextoso-dataset/data/10m'
PATH_save_embedding= '/work/THEIA/CES/oso/hugok/malice/embeddings' #'/work/THEIA/CES/oso/hugok/pastis/malice_data'
PATH_csv_norm='/work/THEIA/CES/oso/hugok/malice/norm_MALICE_s2.csv'
PATH_model_onnx='/work/THEIA/CES/oso/hugok/malice/malice_s2.onnx'

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi
#module load cuda/11.7

echo "Launch python file"

pixi run python src/MALICE.py \
    --data "$data" \
    --path_data "$PATH_data" \
    --path_saved_embeddings "$PATH_save_embedding" \
    --path_csv_norm "$PATH_csv_norm" \
    --path_malice_model "$PATH_model_onnx" \
    --fold ${SLURM_ARRAY_TASK_ID}
echo "End"
