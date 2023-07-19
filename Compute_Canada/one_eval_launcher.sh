#!/bin/bash

#SBATCH --output=/home/maellef/scratch/one_eval.out
#SBATCH --account=rrg-pbellec
#SBATCH --mail-user=maelle.freteault@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=1   
#SBATCH --mem=10G

source /home/maellef/projects/def-pbellec/maellef/finefriends_env/bin/activate
python one_run_eval.py

