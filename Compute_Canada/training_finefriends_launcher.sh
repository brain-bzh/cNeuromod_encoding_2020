#!/bin/bash

#SBATCH --output=/home/maellef/scratch/finetuning_sub01_%x-%j.out
#SBATCH --account=rrg-pbellec
#SBATCH --mail-user=maelle.freteault@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --time=03:00:00
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1   
#SBATCH --mem=30G
#SBATCH --array=1-50

source /home/maellef/projects/def-pbellec/maellef/finefriends_env/bin/activate
sed -n "$SLURM_ARRAY_TASK_ID p" < finetune_subs01_jobs.sh | bash

