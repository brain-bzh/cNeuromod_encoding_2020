#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --mail-user=maelle.freteault@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --array=1-12

source /home/maellef/finetuned_train/bin/activate
sed -n "$SLURM_ARRAY_TASK_ID p" < parcellation_jobs.sh | bash