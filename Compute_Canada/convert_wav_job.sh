#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --mail-user=maelle.freteault@umontreal.ca
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G

python ../audio_utils.py
