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


#---------WIP-----------------------------


#--------END-WIP--------------------------

cmd = dict()
cmd["subject"] = ['01','02']#,'03','04','05','06']
cmd["dataset"] = ["friends"]
cmd["trainData"] = ['s01']
cmd["evalData"] = ['s02']
cmd["sessionsTrain"] = [1]
cmd["sessionsEval"] = [1]
cmd["bs"] = ['30']#["10", "30"]
cmd["ks"] = ['5']#["1", "5"]
cmd["lr"] = ["1e-1", "1e-2"]
cmd["wd"] = ["1e-2", "1e-3"]


cmd["replication"] = ["10"]

# num = "_1"

# cmds = []
# for subject in cmd["subject"]:
#     for dataset in cmd["dataset"]:
#         for session in cmd["session"]:
#             for i_task in range(len(cmd["task"])):
#                 for clusters in cmd["clusters"]:
#                     for states in cmd["states"]:
#                         for batches in cmd["batches"]:
#                             for replications in cmd["replication"]:
#                                 for fwhm in cmd["fwhm"]:
#                                     cmds.append(
#                                         """python  model_training.py -s {sub} -d {dat} -f <None, Bourne, Hidden, Life, Wolf, All> --scale MIST_ROI --nInputs 210 --train100 1.0 --test100 0.5 --val100 0.5 \n""".format(
#                                             sub=subject,
#                                             dat=dataset,
#                                             ses=session,
#                                             tas=cmd["task"][i_task],
#                                             val=cmd["val_task"][i_task],
#                                             batch=batches,
#                                             replication=replications,
#                                             fwhm=fwhm
#                                         )
#                                     )

# with open("cneuromod_embeddings/dypac_jobs{}.sh".format(num), "w") as dy_job:
#     for c in cmds:
#         dy_job.write(c)

# with open("cneuromod_embeddings/dypac_submit_jobs{}.sh".format(num), "r") as dy_sub:
#     lines = dy_sub.readlines()

# for i in range(len(lines)):
#     if "#SBATCH --array=" in lines[i]:
#         lines[i] = "#SBATCH --array=1-{}\n".format(len(cmds))

# with open("cneuromod_embeddings/dypac_submit_jobs{}.sh".format(num), "w") as dy_sub:
#     dy_sub.writelines(lines)