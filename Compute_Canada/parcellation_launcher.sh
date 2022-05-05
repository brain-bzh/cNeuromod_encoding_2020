#!/bin/bash
#SBATCH --output=/home/maellef/scratch/202204_parcellation_%x-%j.out
#SBATCH --account=rrg-pbellec
#SBATCH --time=15:00:00
#SBATCH --mem=12G
#SBATCH --array=1-1

source /home/maellef/projects/def-pbellec/maellef/finefriends_env/bin/activate
sed -n "$SLURM_ARRAY_TASK_ID p" < parcellation_jobs.sh | bash