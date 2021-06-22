#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --mail-user=maelle.freteault@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00
#SBATCH --nodes=1          
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --array=1-100

source /home/maellef/Friends_train/bin/activate
for ((i=0 ; 10 - $i ; i++)); do
    sed -n "$SLURM_ARRAY_TASK_ID p" < early_stopping_bash.sh | bash
done
