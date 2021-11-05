#!/bin/bash

#SBATCH --output=/home/maellef/scratch/%x-%j.out
#SBATCH --account=rrg-pbellec
#SBATCH --mail-user=maelle.freteault@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --time=36:00:00
#SBATCH --nodes=1          
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=1-1000%5

source /home/maellef/finetuned_train/bin/activate
#for ((i=0 ; 10 - $i ; i++)); do
    sed -n "$SLURM_ARRAY_TASK_ID p" < HP_training_Finetuning_P1_jobs_0.sh | bash
#done