import os 

def divide_commands_by_n(file_path, path_name, division_size):
    with open(file_path, "r") as file_data:
        jobs = file_data.readlines()

    division = []
    for i in range(0, len(jobs), division_size):
        i_end = i+division_size if i+division_size < len(jobs) else len(jobs)
        division.append(jobs[i:i_end])

    for i, sub_division in enumerate(division) : 
        with open(path_name+'_{}.sh'.format(i), "w") as new_file:
            new_file.writelines(sub_division)

def select_previous_commands_by_job_state(all_commands, jobID, state = 'TIMEOUT'):
    temp_file = '{}_runs.txt'.format(state)
    cmd = 'sacct --format JobID,State -j {} | grep {} | tee {}'.format(jobID, state, temp_file)
    os.system(cmd)
    
    with open(temp_file, "r") as jobs_data:
        jobs = jobs_data.readlines()
    len_prefix = len('{}_'.format(jobID))
    num_job = [int(job[len_prefix:len_prefix+4]) for job in jobs]
    
    with open(all_commands, "r") as commands_data:
        commands = commands_data.readlines()
    selected_commands = [commands[i-1] for i in num_job]
    
    with open(all_commands+'_{}.sh'.format(state), "w") as new_file:
            new_file.writelines(selected_commands)

if __name__ == "__main__": 
    # pour divide_commands_by_n
    # file_path = './HP_training_FineFriends_sub_3_jobs_0.sh'
    # path_name = './HP_training_FineFriends_sub_3_jobs_0'
    # division_size = 120

    select_previous_commands_by_job_state('HP_training_FineFriends_01_subs_jobs.sh', 5710703, 'TIMEOUT')

