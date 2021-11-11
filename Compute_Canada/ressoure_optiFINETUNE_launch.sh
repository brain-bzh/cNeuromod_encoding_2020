#!/bin/bash
#SBATCH --output=/home/maellef/scratch/%x-%j.out
#SBATCH --account=rrg-pbellec
#SBATCH --time=9:00:00       
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G

source /home/maellef/finetuned_train/bin/activate
python ../model_training.py -s 03 -d friends --trainData s01 s02 --evalData s03 --scale MIST_ROI -f conv5 --lrScheduler --decoupledWD --gpu --wandb