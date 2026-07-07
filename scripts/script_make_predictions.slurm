#!/bin/bash

#SBATCH -J preds
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_preds%j.out
#SBATCH -e script_logs/err_preds%j.err
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

pixi run -e gpu python analyse_results/make_predictions.py \
    --model malice \
    --path_embedding "/work/THEIA/CES/oso/hugok/malice" \
    --path_preds "/work/THEIA/CES/oso/hugok/malice"

echo "End"
