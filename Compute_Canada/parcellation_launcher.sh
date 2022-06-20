#!/bin/bash
#SBATCH --output=/home/maellef/scratch/202205_parcellation_%x-%j.out
#SBATCH --account=rrg-pbellec
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --array=1-1

source /home/maellef/projects/def-pbellec/maellef/finefriends_env/bin/activate
sed -n "$SLURM_ARRAY_TASK_ID p" < parcellation_jobs.sh | bash