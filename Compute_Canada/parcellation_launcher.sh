#!/bin/bash
#SBATCH --output=/home/maellef/scratch/%x-%j.out
#SBATCH --account=rrg-pbellec
#SBATCH --time=12:00:00
#SBATCH --mem=12G
#SBATCH --array=1-12%2

source /home/maellef/finetuned_train/bin/activate
sed -n "$SLURM_ARRAY_TASK_ID p" < parcellation_jobs.sh | bash

