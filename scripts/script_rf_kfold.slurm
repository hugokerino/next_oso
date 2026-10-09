#!/bin/bash

#SBATCH -J rf_kfold
#SBATCH --array=0-44  # 5 folds * 9 eco-zones (indices 0-44),
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 10:00:00
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_rf_kfold_%j.out
#SBATCH -e script_logs/err_rf_kfold_%j.err
#SBATCH --export=none

# Map SLURM_ARRAY_TASK_ID (0-44) -> 5 folds x 9 zones
TASK_ID=${SLURM_ARRAY_TASK_ID}
# fold: 0..4  (each fold has 9 zones)
FOLD=$(( TASK_ID / 9 ))
# zone: 0..8
ZONE=$(( TASK_ID % 9 ))

echo "SLURM_ARRAY_TASK_ID=${TASK_ID} -> FOLD=${FOLD}, ZONE=${ZONE}"

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi

# Create log directory and redirect stdout/stderr at runtime to include fold/zone
LOG_DIR=script_logs
mkdir -p "${LOG_DIR}"
OUT_FILE="${LOG_DIR}/out_rf_kfold_fold${FOLD}_zone${ZONE}.out"
ERR_FILE="${LOG_DIR}/err_rf_kfold_fold${FOLD}_zone${ZONE}.err"

echo "Redirecting stdout -> ${OUT_FILE}  stderr -> ${ERR_FILE}"
# Replace SLURM's static -o/-e logs for per-task logs using shell redirection
exec 1>"${OUT_FILE}" 2>"${ERR_FILE}"

pixi run python RandomForest/main_RF_kfold.py \
    --config_train RandomForest/config_RF.yaml \
    --fold_index ${FOLD} \
    --eco_zone ${ZONE}

echo "End script"
