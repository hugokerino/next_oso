#!/bin/bash

# Launch jobA — LTAE 
id_jobA=$(sbatch scripts/script_main_lightning.slurm \
    --config_train configs/train_trex.yaml \
    --config_model configs/models/LTAE.yaml | awk '{print $NF}')

# Launch jobB
id_jobB=$(sbatch --dependency=afterok:${id_jobA} \
    scripts/script_main_lightning.slurm \
    --config_train configs/train_trex.yaml \
    --config_model configs/models/UTAE.yaml | awk '{print $NF}')

# Launch jobC
id_jobC=$(sbatch --dependency=afterok:${id_jobB} \
    scripts/script_main_lightning.slurm \
    --config_train configs/train_trex.yaml \
    --config_model configs/models/ALISE.yaml | awk '{print $NF}')

echo "ID jobA (LTAE)  : ${id_jobA}"
echo "ID jobB (UTAE)  : ${id_jobB}"
echo "ID jobC (ALISE) : ${id_jobC}"