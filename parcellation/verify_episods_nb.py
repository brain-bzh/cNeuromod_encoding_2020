import os

with open("./parcellation/episodes_list", "r") as dy_sub:
    episodes_list = dy_sub.readlines()

episodes_list = [episode[:episode.find('\n')] if episode.find('\n') > -1 else episode for episode in episodes_list]

path = "/home/maellef/projects/def-pbellec/maellef/data/DataBase/fMRI_Embeddings_fmriprep-2022/auditory_Voxels/friends/"

for sub in os.listdir(path):
    sub_path = os.path.join(path, sub)
    sub_episodes = episodes_list.copy()
    for episode_file in os.listdir(sub_path):
        episode_id = episode_file[-11:-4]
        if episode_id in sub_episodes:
            sub_episodes.remove(episode_id)

    print("for {}, the following episodes are missing : ".format(sub))
    print(sub_episodes)
