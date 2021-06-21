#to do
cmd = dict()
cmd["dataset"] = ["Friends, Movie10"]
cmd["subject"] = [1,2,3,4,5,6]
cmd["session"] = ["all"]
cmd["task"] = ["s01e.[02468]","s01e.[13579]"]
cmd["val_task"] = ["s01e.[13579]","s01e.[02468]"]
cmd["batches"] = ["1"]
cmd["replication"] = ["100"]

num = "_1"

cmds = []
for subject in cmd["subject"]:
    for dataset in cmd["dataset"]:
        for session in cmd["session"]:
            for i_task in range(len(cmd["task"])):
                for clusters in cmd["clusters"]:
                    for states in cmd["states"]:
                        for batches in cmd["batches"]:
                            for replications in cmd["replication"]:
                                for fwhm in cmd["fwhm"]:
                                    cmds.append(
                                        """python  model_training.py -s {sub} -d {dat} -f <None, Bourne, Hidden, Life, Wolf, All> --scale MIST_ROI --nInputs 210 --train100 1.0 --test100 0.5 --val100 0.5 \n""".format(
                                            sub=subject,
                                            dat=dataset,
                                            ses=session,
                                            tas=cmd["task"][i_task],
                                            val=cmd["val_task"][i_task],
                                            batch=batches,
                                            replication=replications,
                                            fwhm=fwhm
                                        )
                                    )

with open("cneuromod_embeddings/dypac_jobs{}.sh".format(num), "w") as dy_job:
    for c in cmds:
        dy_job.write(c)

with open("cneuromod_embeddings/dypac_submit_jobs{}.sh".format(num), "r") as dy_sub:
    lines = dy_sub.readlines()

for i in range(len(lines)):
    if "#SBATCH --array=" in lines[i]:
        lines[i] = "#SBATCH --array=1-{}\n".format(len(cmds))

with open("cneuromod_embeddings/dypac_submit_jobs{}.sh".format(num), "w") as dy_sub:
    dy_sub.writelines(lines)