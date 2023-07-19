import os
from shutil import move

best_models = ['3p9m8alp', '1olpcitm', '2z8swgo9', '2uwpqh5f', 'ydsrklw1', 
'pcmtk3zb', 'o6zr4tpq', 'x3ap6wmq', '1ah0k16g', '28gqq8u1', 
'1i0z4f3a', '3o2u7c9a', '2cx6r4on', '1ial5ec0', '3ppqxglu', 
'yobu6x9u', '2me3xqb7', '3s8anx7r', '3kok1l6h', '352c354f', 
'398eyl79', '22ya72c2', '26wlmmv8', 'm4adsj5c', '3qctm1at', 
'2jl8l7ad', '2n922ju6', 'dcbbxvvo', '2mlgalkf', 'dz4npktm', 
'2n0e0qup', '23695lhb', '10ya4jay','2c201rek', 'e940hqds',
'f0qeh2ll', 'o02gbond', 'qds77g2d', 'uq3y49r0', '2gwx8wxq']

result_path = '/media/maelle/Backup Plus/thèse/Results'
best_model_path = '/home/maelle/Results/converted_models'
os.makedirs(best_model_path, exist_ok=True)
len_id = len('3p9m8alp')
for path, direstories, files in os.walk(result_path):
    for c in files :
        i = c.find('wbid') 
        if i >= 0 : 
            sub = os.path.basename(path)
            new_sub_path = os.path.join(best_model_path, sub)
            os.makedirs(new_sub_path, exist_ok=True)
            id = c[i+4:i+4+len_id]
            if id in best_models:
                print(id)
                actual_path = os.path.join(path, c)
                new_path = os.path.join(new_sub_path, c)
                move(actual_path, new_path)
