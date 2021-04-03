#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --mail-user=maelle.freteault@umontreal.ca
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=30
#SBATCH --mem=64G
#SBATCH --array=1-10

source /home/maellef/Friends_train/bin/activate
sed -n "$SLURM_ARRAY_TASK_ID p" < training_bash.sh | bash
