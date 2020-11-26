
import os

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