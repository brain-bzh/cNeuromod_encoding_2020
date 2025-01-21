#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --mail-user=maelle.freteault@umontreal.ca
#SBATCH --time=24:00:00
#SBATCH --mem=8G

source /home/maellef/finetuned_train/bin/activate
python ../audio_utils.py

