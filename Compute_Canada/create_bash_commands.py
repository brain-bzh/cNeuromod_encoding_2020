# bash command : 

#by default : 
#python  model_training.py -s 01 -d movie10 --trainData wolf --evalData bourne --scale MIST_ROI
#python  model_training.py -s 04 -d friends --trainData s01 s02 s03 --evalData s04 --scale auditory_Voxels

#more args :
#--sessionsTrain 1 -- sessionsVal 2 --select 2 45 8 105 --tr 1.49 --sr 22050
#--bs (batch size) 30 --ks (kernel size) 5
#-f (--finetuneStart) conv5 -o (--outputLayer) conv7
#--patience (early stopping) 15 --delta (early stopping) 0 --train100 0.6 --test100 0.2 --val100 0.2
#--lr (learning rate) 1 --nbepoch 200 --wd (weight decay) 1e-2
#--gpu --decoupledWD --powerTransform --lrScheduler --wandb --comet (True/False options)

exp_name = 'HP_training_Finetuning_P1'
base_cmd = 'python ../model_training.py '
fixed_options = '--lrScheduler --decoupledWD --gpu --wandb\n'

subjects = ['01','02','03','04','05','06']
scales = ['auditory_Voxels', 'MIST_ROI']
select_data = dict()
select_data["dataset"] = ['friends', 'friends', 
                        'movie10', 'movie10', 'movie10', 'movie10'] 
                        #'movie10', 'movie10', 'movie10', 'movie10']
select_data["trainData"] = ['s01', 's02', 
                        'bourne', 'wolf', 'life', 'hidden'] 
                        #'bourne wolf life', 'bourne wolf hidden', 'bourne hidden life', 'wolf hidden life']
select_data["evalData"] = ['s02', 's01', 
                        'bourne', 'wolf', 'life', 'hidden'] 
                        #'hidden', 'life', 'wolf', 'bourne']

lrs = ["1e-1", "1e-2", "1e-3"]
wds = ["1e-2", "1e-3", "1e-4"]
es_patiences = ['10', '15', '20', '30']
es_deltas = ['1e-2', '1e-1', '5e-1']
#finetuneStarts = [None, 'conv7']

#python  model_training.py -s 01 -d movie10 --trainData wolf --evalData bourne --scale MIST_ROI
cmds = []
for subject in subjects : 
    for dataset, trainData, evalData in zip(select_data["dataset"], select_data["trainData"], select_data["evalData"]):
        for scale in scales : 
            for lr in lrs:
                for wd in wds:
                    for patience in es_patiences:
                        for delta in es_deltas:
                            cmd = base_cmd
                            cmd+='-s {} '.format(subject)
                            cmd+='-d {} '.format(dataset)
                            cmd+='--trainData {} '.format(trainData)
                            cmd+='--evalData {} '.format(evalData)
                            cmd+='--scale {} '.format(scale)
                            cmd+='--lr {} '.format(lr)
                            cmd+='--wd {} '.format(wd)
                            cmd+='--patience {} '.format(patience)
                            cmd+='--delta {} '.format(delta)
                            #cmd = cmd+'-f {} '.format(f) if f != None else cmd
                            #cmd = cmd+'--decoupledWD ' if DWD else cmd
                            cmd += fixed_options
                            cmds.append(cmd)

with open("./{}_jobs.sh".format(exp_name), "w") as dy_job:
    for c in cmds:
        dy_job.write(c)

# with open("./training_launcher.sh", "r") as dy_sub:
#     lines = dy_sub.readlines()

# for i in range(len(lines)):
#     if "#SBATCH --array=" in lines[i]:
#         lines[i] = "#SBATCH --array=1-{}\n".format(len(cmds))
#     if 'sed -n "$SLURM_ARRAY_TASK_ID p"' in lines[i]:
#         lines[i] = '    sed -n "$SLURM_ARRAY_TASK_ID p" < {}_jobs.sh | bash'.format(exp_name)

# with open("./training_launcher.sh", "w") as dy_sub:
#     dy_sub.writelines(lines)