import os
import numpy as np
import librosa
from math import floor

def convert_Audio(mediaFile, outFile):
    cmd = 'ffmpeg -i '+mediaFile+' '+outFile
    os.system(cmd)
    return outFile

def load_audio_by_bit(audio, start, end, bitSize, sr=22050, mono=True) : 
    audio_segment = []
    for start_bit in np.arange(start, end, bitSize) : 
        (audio_chunk, _) = librosa.core.load(audio, sr=sr, mono=mono, offset = start_bit, duration = bitSize)
        #To be sure that every audio_chunk have the same size (last audio chunk remove)
        expected_size = floor(sr*bitSize)
        if len(audio_chunk) < expected_size : 
            pass
        else : 
            audio_segment.append(audio_chunk)
    return audio_segment