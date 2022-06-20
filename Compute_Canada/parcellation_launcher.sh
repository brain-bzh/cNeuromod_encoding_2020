#!/bin/bash
#SBATCH --output=/home/maellef/scratch/202206_parcellation_%x-%j.out
#SBATCH --account=def-pbellec
#SBATCH --time=15:00:00
#SBATCH --mem=12G
#SBATCH --array=1-2

module load python/3.8 scipy-stack
source /home/maellef/projects/def-pbellec/maellef/finefriends_env/bin/activate
sed -n "$SLURM_ARRAY_TASK_ID p" < parcellation_jobs.sh | bash