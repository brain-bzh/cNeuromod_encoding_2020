#!/bin/bash

#SBATCH --output=/home/maellef/scratch/prog_finetuning_%x-%j.out
#SBATCH --account=rrg-pbellec
#SBATCH --mail-user=maelle.freteault@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --time=10:00:00
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1        
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --array=1-80%10

source /home/maellef/projects/def-pbellec/maellef/finefriends_env/bin/activate
#for ((i=0 ; 20 - $i ; i++)); do
sed -n "$SLURM_ARRAY_TASK_ID p" < Finetuning_finefriends_jobs.sh | bash
#done