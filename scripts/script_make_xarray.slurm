#!/bin/bash

#SBATCH -J xarray
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_xarray%j.out
#SBATCH -e script_logs/err_xarray%j.err
#SBATCH --export=none

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=14G

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_std
#SBATCH --qos=gpu_all

#SBATCH --time=05:00:00             # Walltime

mkdir -p script_logs

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi

echo "Launch python file"

pixi run -e gpu python analyse_results/make_xarray.py \
    --config_models configs/models/path_to_models.yaml
echo "End"
