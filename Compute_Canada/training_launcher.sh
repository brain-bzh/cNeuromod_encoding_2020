#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --mail-user=maelle.freteault@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --time=36:00:00
#SBATCH --nodes=1          
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=18G
#SBATCH --array=1-1000

source /home/maellef/finetuned_train/bin/activate
#for ((i=0 ; 10 - $i ; i++)); do
    sed -n "$SLURM_ARRAY_TASK_ID p" < HP_training_Finetuning_P1_jobs_0.sh | bash
#done