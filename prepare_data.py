import pandas as pd
import os
import numpy as np
import glob
import csv




import librosa

from scipy.io import wavfile
from tqdm import tqdm
import shutil

import logging
logger = logging.getLogger()


SAMPLERATE = 16000
np.random.seed(2022)


OUT = "/jmain02/home/J2AD003/txk81/wxl53-txk81/MFA_output/corpora/"

def prepare_align(csv_root, OUT, filename):
    data = pd.read_csv(csv_root)
    wav_list = data['wav'].tolist()
    text_list = data['label'].tolist()
    speakers = data['speaker'].tolist()
    num_samples = len(wav_list)

    print(f'preparing {num_samples} samples')
    for i in range(num_samples):
        audio = wav_list[i]
        speaker = speakers[i]
        text = text_list[i]
        temp = audio.split('.')
        ttemp = temp[0].split('/')
        base_name="".join(ttemp[-1:])
        #print(base_name)
        wav, _ = librosa.load(audio, sr = SAMPLERATE)
        wavfile.write(
                    os.path.join(OUT, filename, "{}.wav".format(base_name)),
                    SAMPLERATE,
                    wav.astype(np.int16),
                )
        with open(
            os.path.join(OUT, filename, "{}.lab".format(base_name)),
            "w",
        ) as f1:
            f1.write(text)

def prepare_align_TORGO(csv_root, OUT, filename):
    data = pd.read_csv(csv_root)
    wav_list = data['wav'].tolist()
    text_list = data['label'].tolist()
    speakers = data['speaker'].tolist()
    num_samples = len(wav_list)

    print(f'preparing {num_samples} samples')
    for i in range(num_samples):
        audio = wav_list[i]
        text = text_list[i]
        temp = audio.split('.')
        ttemp = temp[0].split('/')
        base = "".join(ttemp[-3:])
        base_name=f"{ttemp[-4]}_{base}"
        #print(base_name)
        wav, _ = librosa.load(audio, sr = SAMPLERATE)
        wavfile.write(
                    os.path.join(OUT, filename, "{}.wav".format(base_name)),
                    SAMPLERATE,
                    wav.astype(np.int16),
                )
        with open(
            os.path.join(OUT, filename, "{}.lab".format(base_name)),
            "w",
        ) as f1:
            f1.write(text)



# UAS_root = "/jmain02/home/J2AD003/txk81/wxl53-txk81/DASR_2_output/csvs/UAS_corpus/UAS.csv"
# prepare_align(UAS_root, OUT, 'UAS')

# UASC_root = "/jmain02/home/J2AD003/txk81/wxl53-txk81/DASR_2_output/csvs/UAS_corpus/UAS_control.csv"
# prepare_align(UASC_root, OUT, 'UAS_control')

# TOR_root = "/jmain02/home/J2AD003/txk81/wxl53-txk81/DASR_2_output/csvs/TORGO_splits/TORGO.csv"
# prepare_align_TORGO(TOR_root, OUT, 'TORGO')

TORC_root = "/jmain02/home/J2AD003/txk81/wxl53-txk81/DASR_2_output/csvs/TORGO_splits/TORGO_control.csv"
prepare_align_TORGO(TORC_root, OUT, 'TORGO_control')