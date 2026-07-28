#!/bin/bash

#SBATCH -J download_alpha_earth
#SBATCH --array=1-64%64
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 08:00:00
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_download_%A_%a.out
#SBATCH -e script_logs/err_download_%A_%a.err
#SBATCH --export=none

mkdir -p script_logs

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi

YEAR=2020
AOIS_PATH="/home/hkerino/Documents/nextoso-dataset/data/aoi/nextoso_extent.gpkg"
SAVE_PATH="/home/hkerino/Documents/nextoso-dataset/alpha-earth_data"
PROJECT="ee-hugobzhk"

echo "Lancement du téléchargement pour la tâche SLURM array ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"

pixi run python download_embeddings/download_alpha_earth-easy.py \
    --aois_path "$AOIS_PATH" \
    --save_npy_path "$SAVE_PATH" \
    --year "$YEAR" \
    --earth_engine_project "$PROJECT" \
    --array_index ${SLURM_ARRAY_TASK_ID} \
    --n_arrays ${SLURM_ARRAY_TASK_COUNT}

echo "Fin"
