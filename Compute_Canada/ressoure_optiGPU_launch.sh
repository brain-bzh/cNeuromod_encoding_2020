#!/bin/bash
#SBATCH --output=/home/maellef/scratch/%x-%j.out
#SBATCH --account=rrg-pbellec
#SBATCH --time=4:00:00      
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6        
#SBATCH --gres=gpu:1
#SBATCH --mem=4G

source /home/maellef/finetuned_train/bin/activate
python ../model_training.py -s 03 -d friends --trainData s01 s02 --evalData s03 --scale MIST_ROI --lrScheduler --decoupledWD --gpu --wandb