#!/bin/bash

#SBATCH -J testing
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_inferenceV100%j.out
#SBATCH -e script_logs/err_inferenceV100%j.err
#SBATCH --export=none

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=14G

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_std
#SBATCH --qos=gpu_all

#SBATCH --time=01:00:00             # Walltime

mkdir -p script_logs

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi

echo "Launch python file"

$FOLD_INDEX = 0  # For single fold inference, we can set the fold index to 0

pixi run -e gpu python mains/main_supervised_inference.py \
    --config_train configs/models/config_trex.yaml \
    --config_model configs/models/ALISE.yaml \
    --path_dir_logs /work/THEIA/CES/oso/hugok/next_oso/logs/light_training/ALISE_mlp_2026-09-21_kfold \
    --path_save_preds /work/THEIA/CES/oso/hugok/predictions/preds_folder_5 \
    --fold_index $FOLD_INDEX
echo "End"
