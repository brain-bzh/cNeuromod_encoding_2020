import os

def create_dir_if_needed(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def fetchMRI(videofile,fmrilist):
    ### isolate the mkv file (->filename) and the rest of the path (->videopath)

    videopath,filename = os.path.split(videofile)
    #formatting the name to correspond to mri run formatting
    name = filename.replace('_', '')
    if name.startswith('the'):
        name = name.replace('the', '', 1)
    if name.find('life') > -1 :
        name = name.replace('life1', 'life')

    name = name.replace('seg','_run-')
    name = name.replace('subsampl','')
    ## Rename to match the parcellated filenames
    name = name.replace('.wav','npz.npz')

    # list of all parcellated filenames 

    # match videofilename with parcellated files
    mriMatchs = []
    for curfile in fmrilist:
        _, cur_name = os.path.split(curfile)
        if cur_name[23:] == (name):
            mriMatchs.append(curfile)    
    #in case of multiple run for 1 film segment
    name_seg = filename[:-4]

    if len(mriMatchs) > 1 :
        numSessions = []
        for run in mriMatchs :
            index_sess = run.find('ses-vid')
            numSessions.append(int(run[index_sess+7:index_sess+10]))
            
        if numSessions[0] < numSessions[1] : 
            return [(videofile, mriMatchs[0]), (videofile, mriMatchs[1])]

        else : 
            return [(videofile, mriMatchs[1]), (videofile, mriMatchs[0])]
    else :
        return [(videofile, mriMatchs[0])]

def associate_stimuli_with_Parcellation(stimuli_path, path_parcellation):
    stimuli_dic = {}
    for film in os.listdir(stimuli_path):
        film_path = os.path.join(stimuli_path, film)
        if os.path.isdir(film_path):
            film_wav = [os.path.join(film_path, seg) for seg in os.listdir(film_path) if seg[-4:] == '.wav']
            stimuli_dic[film] = sorted(film_wav)

    all_subs = []
    for sub_dir in sorted(os.listdir(path_parcellation)):
        sub_path = os.path.join(path_parcellation, sub_dir)
        all_subs.append([os.path.join(sub_path, mri_data) for mri_data in os.listdir(sub_path) if mri_data[-4:]==".npz"])

    for i, sub in enumerate(all_subs) : 
        sub_segments = {}
        for film, segments in stimuli_dic.items() : 
            sub_segments[film] = []
            for j in range(len(segments)):
                sub_segments[film].extend(fetchMRI(segments[j], sub))

            all_subs[i] = sub_segments

    return all_subs