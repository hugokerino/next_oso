#!/bin/bash

#SBATCH -J testing_supervised_kfold
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_%a_inference_kfoldV100%j.out
#SBATCH -e script_logs/err_%a_inference_kfoldV100%j.err
#SBATCH --export=none

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=14G

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_std
#SBATCH --qos=gpu_all
#SBATCH --time=01:00:00  # Walltime
#SBATCH --array=0-4  # 5 folds (indices 0-4), but SLURM will run in parallel based on available GPUs and user's job limit

mkdir -p script_logs

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi

echo "Launch python file"

pixi run -e gpu python mains/main_supervised_inference.py \
    --config_train configs/supervised_models/config_trex.yaml \
    --config_model configs/supervised_models/ALISE.yaml \
    --path_dir_logs /work/THEIA/CES/oso/hugok/next_oso/logs/light_training/ALISE_mlp_2026-09-21_kfold \
    --path_save_preds /work/THEIA/CES/oso/hugok/predictions/ALISE_preds/v3 \
    --fold_index $SLURM_ARRAY_TASK_ID \
    --measure_time True

echo "End"
