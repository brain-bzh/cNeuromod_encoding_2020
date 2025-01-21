#!/bin/bash

#SBATCH --output=/home/maellef/scratch/finetune_conv1-3_sub01_%x-%j.out
#SBATCH --account=rrg-pbellec
#SBATCH --mail-user=maelle.freteault@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --time=10:00:00
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1   
#SBATCH --mem=30G
#SBATCH --array=1-30%5

module load python/3.8 scipy-stack
source /home/maellef/projects/def-pbellec/maellef/finefriends_env/bin/activate
sed -n "$SLURM_ARRAY_TASK_ID p" < Finetune_FineFriends_01_conv1-3_jobs.sh | bash

