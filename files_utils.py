import os
from audio_utils import convert_Audio

def create_dir_if_needed(path):
    if not os.path.isdir(path):
        os.makedirs(path)

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
            return [(videofile, mriMatchs[0], mriMatchs[1])]#), (videofile, mriMatchs[1])]

        else : 
            return [(videofile, mriMatchs[1], mriMatchs[0])]#), (videofile, mriMatchs[0])]
    elif len(mriMatchs) == 0 : 
        print('no parcellation embedding was found for {}'.format(filename))
        return []
    else :
        return [(videofile, mriMatchs[0])]

def associate_stimuli_with_Parcellation(stimuli_path, path_parcellation, stim_outpath=None):
    stimuli_dic = {}
    for film in os.listdir(stimuli_path):
        film_path = os.path.join(stimuli_path, film)
        if os.path.isdir(film_path):
            if stim_outpath==None:
                film_wav = [os.path.join(film_path, seg) for seg in os.listdir(film_path) if seg[-4:] == '.wav']
            else : 
                #if outpath, you need to create wav from mkv    
                film_mkv = [os.path.join(film_path, seg) for seg in os.listdir(film_path) if seg[-4:] == '.mkv']
                film_wav = [os.path.join(stim_outpath, seg[:-4]+'.wav') for seg in os.listdir(film_path) if seg[-4:] == '.mkv']
                for mkv, wav in zip(film_mkv, film_path):
                    convert_Audio(mkv, wav)

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

def cNeuromod_subject_convention(path, name, zero_index = True):
    num = int(name[-1])
    if zero_index : 
        num +=1

    new_name = 'sub'+str(num)
    if new_name != name:
        prev_path = os.path.join(path, name)
        new_path = os.path.join(path, new_name)
        os.rename(prev_path, new_path)

def rename_object(path, keyword_to_replace, rename_convention, objects=['dirs','files']):
    for path, dirs, files in os.walk(path):
        for key_object in objects :
            if key_object == 'dirs':
                key_list = dirs
            elif key_object == 'files':
                key_list = files
            
            for obj in key_list:
                #print(obj)
                if keyword_to_replace in obj:
                    rename_convention(path, obj)

if __name__ == "__main__":
    path = "/home/maelle/Results/20210201_tests_kernel_voxel_Norm_embed2020"
    rename_object(path, 'subject_', cNeuromod_subject_convention, objects=['dirs'])

