import os
from pandas import isnull
from audio_utils import convert_Audio

all_films = {
    'bourne':'bourne_supremacy',
    'wolf':'wolf_of_wall_street',
    'life':'life',
    'figures':'hidden_figures',
}

def print_dict(dico):
    for key, value in dico.items():
        print(key)    
        for item in value :
            print('      {}'.format(item))

def extract_value_from_string(string, start_index, stop_condition=(lambda x: False)):
    temp=''
    condition = False
    while not condition:
        try:
            temp += string[start_index]
            if '.' not in temp:
                target_value = int(temp)
            else :
                target_value = float(temp)

            start_index+=1
            condition = stop_condition(string[start_index])
        except (IndexError, ValueError):
            break

    return target_value

def result_name(dataset, scale, model, batchsize, kernel_size, patience_es, delta_es, learning_rate, weigth_decay, 
                            decoupled_wd, lr_scheduler, power_transform, finetune_start, wandb_id, no_training, no_init, 
                            year = 2022):
    
    outfile_name = '{}_{}_{}_'.format(dataset, scale, model)
    outfile_name +='{:03}{:02}{:02}'.format(batchsize, kernel_size, patience_es)
    if year == 2021 :
        outfile_name +='{:.0e}'.format(delta_es)[-3:]+'{:.0e}'.format(learning_rate)[-3:]+'{:.0e}'.format(weigth_decay)[-3:]
    else : 
        outfile_name +='_{:.0e}'.format(delta_es)+'_{:.0e}'.format(learning_rate)+'_{:.0e}'.format(weigth_decay)
    outfile_name += '_opt'
    outfile_name = outfile_name+'1' if decoupled_wd else outfile_name+'0'
    outfile_name = outfile_name+'1' if lr_scheduler else outfile_name+'0'
    outfile_name = outfile_name+'1' if power_transform else outfile_name+'0'
    outfile_name = outfile_name+'_NoTrain' if no_training else outfile_name
    outfile_name = outfile_name+'_NoInit' if no_init else outfile_name
    if finetune_start != None or not isnull(finetune_start) : 
        outfile_name = outfile_name+'_f_'+finetune_start
    outfile_name += '_wbid'+format(wandb_id)
    return outfile_name

def fetchMRI(videofile,fmrilist):
    ### isolate the mkv file (->filename) and the rest of the path (->videopath)
    filmPath,filename = os.path.split(videofile)
    datasetPath, film = os.path.split(filmPath)
    _, dataset = os.path.split(datasetPath)
    task, _  = os.path.splitext(filename[len(dataset)+1:])

    mriMatchs = []
    for curfile in fmrilist:
        _, cur_name = os.path.split(curfile)
        #print(task, cur_name)
        if cur_name.find(task) >-1 :
            mriMatchs.append(curfile)    

    if len(mriMatchs) > 1 :
        numSessions = []
        for run in mriMatchs :
            index_ses = run.find('ses-')
            numSessions.append(int(run[index_ses+4:index_ses+7]))
        if numSessions[0] < numSessions[1] :
            return [(videofile, mriMatchs[0], mriMatchs[1])]#), (videofile, mriMatchs[1])]
        else : 
            return [(videofile, mriMatchs[1], mriMatchs[0])]#), (videofile, mriMatchs[0])]
    elif len(mriMatchs) == 0 : 
        print('no parcellation embedding was found for {}'.format(filename))
        return []
    else :
        return [(videofile, mriMatchs[0])]

def associate_stimuli_with_Parcellation(stimuli_path, path_parcellation):
    stimuli_wav = [os.path.join(stimuli_path, seg) for seg in sorted(os.listdir(stimuli_path)) if seg[-4:] == '.wav']
    parcellation_list = [os.path.join(path_parcellation, mri_data) for mri_data in sorted(os.listdir(path_parcellation)) if mri_data[-4:]==".npz"]
    pair_wav_mri = []
    for wav in stimuli_wav:
        pair_wav_mri.extend(fetchMRI(wav, parcellation_list))
    return pair_wav_mri

def cNeuromod_subject_convention(path, name, zero_index = True):
    num = int(name[-1])
    if zero_index : 
        num +=1

    return 'sub'+str(num)

def cNeuromod_stimuli_convention(path, name):
    #name_model : sub-0X_ses-XXX_task-<sXXe/film><xxx>.wav
    #path_model : your_path/DataBase/stimuli/dataset/<films, seasons>

    name, ext = os.path.splitext(name)
    film = os.path.basename(path)
    dataset = os.path.basename(os.path.dirname(path))

    run = name[-4:] if dataset == 'friends' else name[-2:]

    new_name = dataset+'_'+film+run+ext
    return new_name

def cNeuromod_embeddings_convention(path, name):
    #name_model : sub-0X_ses-XXX_task-<sXXe/film><xxx>.npz
    #path_model : your_path/DataBase/fMRI_Embeddings/your_embedding/dataset/subject

    _, ext = os.path.splitext(name)
    subject = os.path.basename(path)

    sessIndex = name.find('ses-')+4
    vidIndex = name.find('vid')+3
    if sessIndex > 3 or vidIndex > 2 :
        i = max(sessIndex, vidIndex)
        session = 'ses-'+name[i:i+3]
    else : 
        print('error : no sufficient information to name the file')
        return None
    
    dataset = os.path.basename(os.path.dirname(path))
    if dataset == 'movie10':
        for film in all_films.keys():
            if name.find(film)>-1:
                break
        taskNumI = name.find('run-')+4
        task = 'task-'+film+name[taskNumI:taskNumI+2]

        new_name = subject+'_'+session+'_'+task+ext

    elif dataset == 'friends':
        new_name = name
    
    return new_name

def rename_objects_in_dataset(dataset_path, keyword_to_replace, rename_convention, objects=['dirs','files']):
    for path, dirs, files in os.walk(dataset_path):
        for key_object in objects :
            if key_object == 'dirs':
                key_list = dirs
            elif key_object == 'files':
                key_list = files
            
            for filename in key_list:
                if keyword_to_replace in filename:
                    new_name = rename_convention(path, filename)
                    prev_path = os.path.join(path, filename)
                    new_path = os.path.join(path, new_name)
                    os.rename(prev_path, new_path)


if __name__ == "__main__":
    path_vox = "/home/maelle/DataBase/stimuli/friends"
    rename_objects_in_dataset(path_vox, 'friends_s', cNeuromod_stimuli_convention, objects=['files'])
