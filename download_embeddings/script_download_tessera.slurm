#!/bin/bash

#SBATCH -J test_download_tessera
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-cpu=16G
#SBATCH -t 3:00:00
#SBATCH -A theia_oso
#SBATCH -o logs/out_tessera_%A_%a.out
#SBATCH -e logs/err_tessera_%A_%a.err
#SBATCH --export=none
#SBATCH --array=1-64%64

mkdir -p logs

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi"
module load pixi

PATH_AOI="/work/THEIA/CES/oso/hugok/pastis/metadata.geojson"
PATH_OUTPUT="/work/THEIA/CES/oso/hugok/tessera/tessera_data_pastis"

pixi run python download_tessera.py \
    --aois_path $PATH_AOI \
    --save_path $PATH_OUTPUT
