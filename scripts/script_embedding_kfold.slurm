#!/bin/bash

#SBATCH -J emb_kfold
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_emb%j.out
#SBATCH -e script_logs/err_emb%j.err
#SBATCH --export=none

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=14G
#SBATCH --array=0-4  # 5 folds (indices 0-4), but SLURM will run up to 2 in parallel based on available GPUs

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_std
#SBATCH --qos=gpu_all

#SBATCH --time=10:00:00            # Walltime

#SBATCH --signal=SIGUSR1@90         # For auto-resubmit

mkdir -p script_logs

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi

echo "Launch python file"

task_id=${SLURM_ARRAY_TASK_ID:-0}

# Load folds from YAML configuration
fold_config=$(pixi run -e gpu python scripts/get_folds_from_yaml.py $task_id)
IFS='|' read -r folds_train_selected folds_val_selected folds_test_selected <<< "$fold_config"
read -r -a folds_train_array <<< "$folds_train_selected"
read -r -a folds_val_array <<< "$folds_val_selected"
read -r -a folds_test_array <<< "$folds_test_selected"

pixi run -e gpu python mains/main_embedding_kfold.py \
    --data "next_oso" \
    --model "malice" \
    --decoder_type "mlp" \
    --path_embedding "/work/THEIA/CES/oso/hugok/malice" \
    --path_label "/work/THEIA/CES/oso/hugok/labels" \
    --path_mapping_folder "/work/THEIA/CES/oso/hugok/s2_data" \
    --path_logs "/work/THEIA/CES/oso/hugok/next_oso/logs" \
    --path_name_class "/work/THEIA/CES/oso/hugok/files_s2/oso_classes.csv" \
    --no-preload_data \
    --batch_size 8 \
    --num_worker 6 \
    --max_epochs 1000 \
    --label_type "max_class" \
    --folds_train "${folds_train_array[@]}" \
    --folds_val "${folds_val_array[@]}" \
    --folds_test "${folds_test_array[@]}" \
    --fold_index "${task_id}"

echo "End"
