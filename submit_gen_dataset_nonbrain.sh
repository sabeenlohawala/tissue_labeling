#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -c 10
#SBATCH --mem-per-cpu=5G # per node memory
#SBATCH -p gablab
#SBATCH -o ./logs/gen_dataset_fs.out
#SBATCH -e ./logs/gen_dataset_fs.err

export PATH="/om2/user/sabeen/miniconda/envs/tissue_labeling/bin/:$PATH"
conda init bash
# source activate tissue_labeling
echo "Submitted Job: $SLURM_JOB_ID"

# srun python -u scripts/gen_dataset_nonbrain.py /om/scratch/tmp/sabeen/kwyk_chunk/ 9 /om/scratch/Fri/sabeen/kwyk_h5_nonbrains/ --find_matthias_filter 0
# srun python -u scripts/gen_dataset_nonbrain.py /om/scratch/tmp/sabeen/kwyk_chunk/ 9 /om/scratch/Fri/sabeen/kwyk_h5_matthias_2/ --find_matthias_filter 1
srun python -u scripts/gen_dataset_nonbrain.py /om/scratch/tmp/sabeen/kwyk_chunk/ 9 /om2/user/sabeen/feature_sums/ --find_matthias_filter 2