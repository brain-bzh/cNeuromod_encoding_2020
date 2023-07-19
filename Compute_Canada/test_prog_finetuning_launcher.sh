#!/bin/bash

#SBATCH --output=/home/maellef/scratch/prog_finetuning_%x-%j.out
#SBATCH --account=rrg-pbellec
#SBATCH --mail-user=maelle.freteault@umontreal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --time=05:00:00
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1        
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --array=1

source /home/maellef/projects/def-pbellec/maellef/finefriends_env/bin/activate
python ../model_training.py -s 03 -d friends --trainData s01 s02 s03 --evalData s04 --scale MIST_ROI -f 'conv4' -e 10 --lr 1e-4 --wd 1e-3 --patience 15 --delta 0.5 --bs 70 --ks 5 --lrScheduler --decoupledWD --gpu --wandb