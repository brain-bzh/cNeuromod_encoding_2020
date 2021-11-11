#!/bin/bash

#SBATCH --output=/home/maellef/scratch/%x-%j.out
#SBATCH --account=rrg-pbellec
#SBATCH --mail-user=maelle.freteault@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --time=15:00:00
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4        
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --array=1-120%10

source /home/maellef/finetuned_train/bin/activate
#for ((i=0 ; 10 - $i ; i++)); do
sed -n "$SLURM_ARRAY_TASK_ID p" < HP_training_FineFriends_sub_3_jobs_0_0.sh | bash
#done