file_path = './HP_training_FineFriends_sub_3_jobs_0.sh'
path_name = './HP_training_FineFriends_sub_3_jobs_0'
division_size = 120

with open(file_path, "r") as file_data:
    jobs = file_data.readlines()

division = []
for i in range(0, len(jobs), division_size):
    i_end = i+division_size if i+division_size < len(jobs) else len(jobs)
    division.append(jobs[i:i_end])

for i, sub_division in enumerate(division) : 
    with open(path_name+'_{}.sh'.format(i), "w") as new_file:
        new_file.writelines(sub_division)
