python ../model_training.py -s 03 -d friends --trainData s01 s02 s03 --evalData s04 --scale auditory_Voxels --lr 1e-4 --wd 1e-2 --patience 15 --delta 0.1 --bs 70 --ks 5 --lrScheduler --decoupledWD --gpu --wandb
python ../model_training.py -s 03 -d friends --trainData s01 s02 s03 --evalData s04 --scale MIST_ROI --lr 1e-4 --wd 1e-3 --patience 15 --delta 0.5 --bs 70 --ks 5 --lrScheduler --decoupledWD --gpu --wandb
python ../model_training.py -s 03 -d friends --trainData s01 s02 s03 --evalData s04 --scale auditory_Voxels -f 'conv7' --lr 1e-4 --wd 1e-2 --patience 15 --delta 0.1 --bs 70 --ks 5 --lrScheduler --decoupledWD --gpu --wandb
python ../model_training.py -s 03 -d friends --trainData s01 s02 s03 --evalData s04 --scale MIST_ROI -f 'conv7' --lr 1e-4 --wd 1e-3 --patience 15 --delta 0.5 --bs 70 --ks 5 --lrScheduler --decoupledWD --gpu --wandb
python ../model_training.py -s 03 -d friends --trainData s01 s02 s03 --evalData s04 --scale auditory_Voxels -f 'conv6' --lr 1e-4 --wd 1e-2 --patience 15 --delta 0.1 --bs 70 --ks 5 --lrScheduler --decoupledWD --gpu --wandb
python ../model_training.py -s 03 -d friends --trainData s01 s02 s03 --evalData s04 --scale MIST_ROI -f 'conv6' --lr 1e-4 --wd 1e-3 --patience 15 --delta 0.5 --bs 70 --ks 5 --lrScheduler --decoupledWD --gpu --wandb
python ../model_training.py -s 03 -d friends --trainData s01 s02 s03 --evalData s04 --scale auditory_Voxels -f 'conv5' --lr 1e-4 --wd 1e-2 --patience 15 --delta 0.1 --bs 70 --ks 5 --lrScheduler --decoupledWD --gpu --wandb
python ../model_training.py -s 03 -d friends --trainData s01 s02 s03 --evalData s04 --scale MIST_ROI -f 'conv5' --lr 1e-4 --wd 1e-3 --patience 15 --delta 0.5 --bs 70 --ks 5 --lrScheduler --decoupledWD --gpu --wandb
python ../model_training.py -s 03 -d friends --trainData s01 s02 s03 --evalData s04 --scale auditory_Voxels -f 'conv4' --lr 1e-4 --wd 1e-2 --patience 15 --delta 0.1 --bs 70 --ks 5 --lrScheduler --decoupledWD --gpu --wandb
python ../model_training.py -s 03 -d friends --trainData s01 s02 s03 --evalData s04 --scale MIST_ROI -f 'conv4' --lr 1e-4 --wd 1e-3 --patience 15 --delta 0.5 --bs 70 --ks 5 --lrScheduler --decoupledWD --gpu --wandb


 #MIST ROI : bs = 70, ks = 5, lr = 0.0001, wd = 0.001, patience = 15, delta = 0.5
 #voxels : bs = 70, ks = 5, lr = 0.0001, wd = 0.01, patience = 15, delta = 0.1
 #cmd = cmd+'-f {} '.format(f) if f != None else cmd