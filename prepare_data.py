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

def wav_txt_lst(root):
   wav_lst = glob.glob(os.path.join(root, "**/*.wav"), recursive=True)
   text_lst = glob.glob(os.path.join(root, "**/*.txt"), recursive=True)
   print("tot wav audio files {}".format(len(wav_lst)))
   print("tot txt files {}".format(len(text_lst)))
   return wav_lst, text_lst

def UAS_labels(UAS_p):
    #labels = pd.read_csv(f"{UAS_p}UA_speech_labels.csv")   #specify path to UAS CSV file here
    labels = pd.read_csv("/fastdata/acr22wl/UAS/UA_speech_labels.csv")

    labels_dict = dict(zip(labels['FILE NAME'], labels['WORD']))
    for k,v in labels_dict.items():
        labels_dict.update({k: v.upper()})
    return labels_dict

def TOR_labels(text_files):
   text_labels = {}
   punc = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
   for txtf in sorted(text_files):
      ttag = txtf.split('.')
      temp = " ".join(ttag[0:-1])
      ttemp = temp.split('/')
      tag = "".join(ttemp[0:-2])  # contentF03Session1
      tag2 = tag + ttemp[-1] # contentF03Session10011
      with open(txtf, "r") as f:
         lines = f.readlines()
      for i, line in enumerate(lines):
         line = line.strip("\n")
         line = line.translate(str.maketrans('', '', punc))
         line = line.upper()
         text_labels[tag2] = line
   return text_labels

def create_lab_file(wav_filepath, transcript):
    # Get the directory containing the WAV file
    directory = os.path.dirname(wav_filepath)

    # Create the .lab file path
    lab_filepath = os.path.join(directory, os.path.splitext(os.path.basename(wav_filepath))[0] + '.lab')

    # Write transcript to .lab file
    with open(lab_filepath, 'w') as lab_file:
        lab_file.write(transcript)

def make_csv(wav_lst, labels_dict, OUT, filename):
    os.makedirs(f"{OUT}UAS/", exist_ok=True)
    output_csv_filename=f"{OUT}{filename}.csv"
    labels = {}
    data = []
    headers = ['wav', 'speaker', 'block', 'tag', 'ID', 'mic', 'label']
    for audio in sorted(wav_lst):
        temp_a = audio.split('/')
        spkr = temp_a[-2]
        temp_b = audio.split('_')
        opt_1 = temp_b[2]
        opt_2 = temp_b[1] + '_' + temp_b[2]
        block = temp_b[1]
        mic = temp_b[-1]
        if opt_1 in labels_dict:
            labels[audio] = labels_dict[opt_1]
            text_label = (labels_dict[opt_1])
            #tag = opt_1
        else:
            labels[audio] = labels_dict[opt_2]
            text_label = (labels_dict[opt_2])
            #tag = opt_2
        data.append([audio, f"UAS_{spkr}", block, opt_1, f"UAS_{spkr}_{block}_{opt_1}_{mic}", mic, text_label])
    with open(output_csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)  # Write the header row
        csv_writer.writerows(data)
    #df = pd.read_csv(output_csv_filename)
    with open(output_csv_filename, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            wav_filepath = row['wav']
            transcript = row['label']
            create_lab_file(wav_filepath, transcript)


def make_csv_TOR(wav_lst, text_labels, OUT, filename):
    os.makedirs(f"{OUT}Torgo/", exist_ok=True)
    output_csv_filename=f"{OUT}{filename}.csv"
    headers = ['wav', 'speaker', 'ID', 'mic', 'label']
    data = []
    for audio in sorted(wav_lst):
        ttag = audio.split('.')
        temp = " ".join(ttag[0:-1])
        ttemp = temp.split('/')
        spkr = ttemp[4]
        m = ttemp[6].split('_')
        mic = m[-1]
        w = ttemp[-1]
        tag = "".join(ttemp[0:-2])  # contentF03Session1
        tag2 = tag + ttemp[-1]
        label = text_labels[tag2]
        data.append([audio, f"TOR_{spkr}", f"TOR_{spkr}_{mic}_{w}", "mic", label])
    with open(output_csv_filename, 'w', newline='') as csvfile:
      csv_writer = csv.writer(csvfile)
      csv_writer.writerow(headers)  # Write the header row
      csv_writer.writerows(data)
    #df = pd.read_csv(output_csv_filename)
    with open(output_csv_filename, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            wav_filepath = row['wav']
            transcript = row['label']
            create_lab_file(wav_filepath, transcript)


UAS_root = "/fastdata/acr22wl/UAS/dysarthria"
UASC_root = "/fastdata/acr22wl/UAS/control"

OUT = "/fastdata/acr22wl/MFA_output/corpora"

labels_dict = UAS_labels(UAS_root)

wav_lst, txt_lst = wav_txt_lst(UAS_root)
make_csv(wav_lst, labels_dict, OUT, 'UAS')

wav_lst, txt_lst = wav_txt_lst(UASC_root)
make_csv(wav_lst, labels_dict, OUT, 'UAS_control')

root = "/fastdata/acr22wl/Torgo_use"
root_con = "/fastdata/acr22wl/Torgo_control"

wav_lst, txt_lst = wav_txt_lst(root)
text_labels = TOR_labels(txt_lst)
make_csv_TOR(wav_lst, text_labels, OUT, 'Torgo')

wav_lst, txt_lst = wav_txt_lst(root_con)
text_labels = TOR_labels(txt_lst)
make_csv_TOR(wav_lst, text_labels, OUT, 'Torgo_control')