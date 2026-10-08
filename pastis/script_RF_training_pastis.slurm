#!/bin/bash

#SBATCH -J rf_pastis_kfold
#SBATCH --array=0-19  # 5 folds * 4 tiles (indices 0-19),
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 10:00:00
#SBATCH -A theia_oso
#SBATCH -o script_logs/out_rf_pastis_kfold_%j.out
#SBATCH -e script_logs/err_rf_pastis_kfold_%j.err
#SBATCH --export=none

# Map SLURM_ARRAY_TASK_ID (0-19) -> 5 folds x 4 tiles
TASK_ID=${SLURM_ARRAY_TASK_ID}
# fold: 0..4  (each fold has 4 tiles)
FOLD=$(( TASK_ID / 4 ))
# tile: 0..3
TILE=$(( TASK_ID % 4 ))

echo "SLURM_ARRAY_TASK_ID=${TASK_ID} -> FOLD=${FOLD}, TILE=${TILE}"

cd $SLURM_SUBMIT_DIR
echo -e "\nRépertoire courant : $(pwd)"

echo "Load pixi envs"
module purge
module load pixi

# Create log directory and redirect stdout/stderr at runtime to include fold/zone
LOG_DIR=script_logs
mkdir -p "${LOG_DIR}"
OUT_FILE="${LOG_DIR}/out_rf_PASTIS_kfold_fold${FOLD}_zone${TILE}.out"
ERR_FILE="${LOG_DIR}/err_rf_PASTIS_kfold_fold${FOLD}_zone${TILE}.err"

echo "Redirecting stdout -> ${OUT_FILE}  stderr -> ${ERR_FILE}"
# Replace SLURM's static -o/-e logs for per-task logs using shell redirection
exec 1>"${OUT_FILE}" 2>"${ERR_FILE}"

pixi run python pastis/main_RF_training_pastis.py \
    --config_train configs/RF_config.yaml \
    --fold_index ${FOLD} \
    --tile ${TILE}

echo "End script"
