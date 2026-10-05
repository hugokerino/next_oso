#!/bin/bash

#SBATCH -J emb_testing_kfold
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_embk%j.out
#SBATCH -e script_logs/err_embk%j.err
#SBATCH --export=none

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=14G
#SBATCH --array=0-4  # 5 folds (indices 0-4), but SLURM will run up to 2 in parallel based on available GPUs

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_std
#SBATCH --qos=gpu_all

#SBATCH --time=1:00:00            # Walltime

#SBATCH --signal=SIGUSR1@90         # For auto-resubmit

mkdir -p script_logs

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi

echo "Launch python file"

task_id=${SLURM_ARRAY_TASK_ID:-0}


pixi run -e gpu python mains/main_embedding_testing.py \
    --config /work/THEIA/CES/oso/hugok/next_oso/configs/FM_models/config_trex.yaml \
    --path_dir_logs /work/THEIA/CES/oso/hugok/next_oso/logs/alpha-earth/mlp_2026-09-08_kfold \
    --fold_index $SLURM_ARRAY_TASK_ID  # Pass the fold index from the array

echo "End"
