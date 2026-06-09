#!/bin/bash

#SBATCH -J alphaearth
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_ae%j.out
#SBATCH -e script_logs/err_ae%j.err
#SBATCH --export=none

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=7G

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --qos=gpu_all

#SBATCH --time=05:00:00            # Walltime

#SBATCH --signal=SIGUSR1@90         # For auto-resubmit

mkdir -p script_logs

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi

echo "Launch python file"

pixi run -e gpu python mains/main_alpha_earth.py \
    --path_embedding "/work/THEIA/CES/oso/hugok/alpha_earth_data" \
    --path_label "/work/THEIA/CES/oso/hugok/labels" \
    --path_mapping_folder "/work/THEIA/CES/oso/hugok/s2_data" \
    --path_logs "/work/THEIA/CES/oso/hugok/next_oso/logs/alpha-earth" \
    --path_name_class "/work/THEIA/CES/oso/hugok/files_s2/oso_classes.csv" \
    --batch_size 64 \
    --num_worker 15 \
    --max_epochs 1000 \
    --preload_data True \

echo "End"
