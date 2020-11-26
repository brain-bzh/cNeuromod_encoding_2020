import os
from audio_utils import convert_Audio

stimuli_path = '/home/brain/Data_Base/cneuromod/movie10/stimuli' #'/home/maelle/Database/cneuromod/movie10/stimuli'
for film in os.listdir(stimuli_path):
    film_path = os.path.join(stimuli_path, film)
    if os.path.isdir(film_path):
        for seg in os.listdir(film_path):
             if seg[-4:] == '.mkv':
                 seg_path = os.path.join(film_path, seg)
                 print(seg_path)
                 outfile = seg_path[:-4]+'.wav'
                 print(outfile)
                 convert_Audio(seg_path, outfile)