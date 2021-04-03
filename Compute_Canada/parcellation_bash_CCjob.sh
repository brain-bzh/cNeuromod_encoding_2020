#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --mail-user=maelle.freteault@umontreal.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --array=1-5

sed -n "$SLURM_ARRAY_TASK_ID p" < bash_jobs.sh | bash