#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --mail-user=maelle.freteault@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --array=1-12

source /home/maellef/finetuned_train/bin/activate
sed -n "$SLURM_ARRAY_TASK_ID p" < parcellation_jobs.sh | bash