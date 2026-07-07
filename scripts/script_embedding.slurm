#!/bin/bash

#SBATCH -J malice
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_emb%j.out
#SBATCH -e script_logs/err_emb%j.err
#SBATCH --export=none

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=14G

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_std
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

pixi run -e gpu python mains/main_embedding.py \
    --model "malice" \
    --time_feature \
    --path_embedding "/work/THEIA/CES/oso/hugok/malice" \
    --path_label "/work/THEIA/CES/oso/hugok/labels" \
    --path_mapping_folder "/work/THEIA/CES/oso/hugok/s2_data" \
    --path_logs "/work/THEIA/CES/oso/hugok/next_oso/logs/malice" \
    --path_name_class "/work/THEIA/CES/oso/hugok/files_s2/oso_classes.csv" \
    --no-preload_data \
    --batch_size 8 \
    --num_worker 6 \
    --max_epochs 1000 \
    --label_type "max_class"

echo "End"
