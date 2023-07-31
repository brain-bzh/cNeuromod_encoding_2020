#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --output=/home/maellef/projects/def-pbellec/maellef/projects/cNeuromod_encoding_2020/AssWandbResults_%x-%j.out
#SBATCH --mail-user=maelle.freteault@umontreal.ca
#SBATCH --time=00:30:00
#SBATCH --mem=2G

source /home/maellef/projects/def-pbellec/maellef/finefriends_env/bin/activate
python ../ordering_wandb_result_file.py
