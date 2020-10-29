import os

def search_directory(path, dir_name, look_in_hidden_dir = False):
'''
Have to see if pybids isn't easier
to finish; instead of giving the exact path, this function look for one specified directory given the name and the starting path to search; 
'''
    sub_files = os.listdir(path)
    dirs_path = [(subfile, os.path.join(path, subfile))for subfile in sub_files if os.path.isdir(os.path.join(path, subfile)) and not subfile[0]=='.']
    stim_path = ''
    
    for name, sub_path in dirs_path :
        if name == dir_name:
            print(sub_path)
            return sub_path
        else : 
            func_1(sub_path)
        


if __name__ == "__main__":
    path = '/media/brain/Elec_HD/cneuromod'
    stim_path = func_1(path)
    print(stim_path)
