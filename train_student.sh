#!/bin/sh
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e

# Activate your conda environment
CONDA_ENV_NAME=torchdistill
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}


# ==============================
# Finally, run the experiment!
# ==============================
# Read line number ${SLURM_ARRAY_TASK_ID} from the experiment file and run it
# ${SLURM_ARRAY_TASK_ID} is simply the number of the job within the array. If
# you execute `sbatch --array=1:100 ...` the jobs will get numbers 1 to 100
# inclusive.

torchrun torchdistill/examples/image_classification.py --config torchdistill/configs/sample/cifar100/ce/resnet44-teacher-50k.yaml --log log/cifar100/ce/resnet44-teacher-50k.log --eval_period 10

# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
