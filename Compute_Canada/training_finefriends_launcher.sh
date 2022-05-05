#!/bin/bash

#SBATCH --output=/home/maellef/scratch/finetuning_sub2_%x-%j.out
#SBATCH --account=rrg-pbellec
#SBATCH --mail-user=maelle.freteault@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --time=30:00:00
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2   
#SBATCH --mem=30G
#SBATCH --array=1-2%2

source /home/maellef/projects/def-pbellec/maellef/finefriends_env/bin/activate
sed -n "$SLURM_ARRAY_TASK_ID p" < finetune_subs2_jobs_TIMEOUT.sh | bash

