#!/bin/bash

#SBATCH -J testing_kfold
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_testing_kfoldV100%j.out
#SBATCH -e script_logs/err_testing_kfoldV100%j.err
#SBATCH --export=none

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=14G

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_std
#SBATCH --qos=gpu_all
#SBATCH --time=01:00:00             # Walltime
#SBATCH --array=0-4  # 5 folds (indices 0-4), but SLURM will run up to 2 in parallel based on available GPUs

mkdir -p script_logs

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi

echo "Launch python file"

pixi run -e gpu python mains/main_testing_kfold.py \
    --config_train configs/models/train_trex.yaml \
    --config_model configs/models/ALISE.yaml \
    --path_dir_log /work/THEIA/CES/oso/hugok/next_oso/light_training/ALISE_kfold_spatial_07-17 \
    --fold_index $SLURM_ARRAY_TASK_ID

echo "End"
