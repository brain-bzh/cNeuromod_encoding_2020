#!/bin/bash

#SBATCH --output=/home/maellef/scratch/one_eval_%j.out
#SBATCH --account=rrg-pbellec
#SBATCH --mail-user=maelle.freteault@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=1   
#SBATCH --mem=20G
#SBATCH --array=1-36%12

module load python/3.8 scipy-stack
source /home/maellef/projects/def-pbellec/maellef/finefriends_env/bin/activate
sed -n "$SLURM_ARRAY_TASK_ID p" < ./Compute_Canada/one_eval_jobs.sh | bash

