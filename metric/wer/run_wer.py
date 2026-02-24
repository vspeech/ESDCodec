import sys, os
import inspect
import whisper
from tqdm import tqdm
import multiprocessing
from jiwer import compute_measures
from zhon.hanzi import punctuation
import string
import numpy as np
import soundfile as sf
import scipy
import random
import torch
import zhconv
#from funasr import AutoModel

punctuation_all = punctuation + string.punctuation

wav_res_text_path = sys.argv[1]
res_path = sys.argv[2]
lang = sys.argv[3] # en
device = "cuda:0"

def load_en_model():
    model = whisper.load_model("large-v3")
    model = model.to(device)
    return model

def process_one(hypo, truth):
    raw_truth = truth
    raw_hypo = hypo

    for x in punctuation_all:
        truth = truth.replace(x, '')
        hypo = hypo.replace(x, '')

    truth = truth.replace('  ', ' ')
    hypo = hypo.replace('  ', ' ')

    truth = truth.lower()
    hypo = hypo.lower()
    
    measures = compute_measures(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures["wer"]
    subs = measures["substitutions"] / len(ref_list)
    dele = measures["deletions"] / len(ref_list)
    inse = measures["insertions"] / len(ref_list)
    return (truth, hypo, wer, subs, dele, inse)


def run_asr(wav_res_text_path, res_path):
    model = load_en_model()

    params = []
    for line in open(wav_res_text_path).readlines():
        line = line.strip()
        if len(line.split('|')) == 2:
            wav_res_path, text_ref = line.split('|')
        elif len(line.split('|')) == 3:
            wav_res_path, wav_ref_path, text_ref = line.split('|')
        elif len(line.split('|')) == 4: # for edit
            wav_res_path, _, text_ref, wav_ref_path = line.split('|')
        else:
            raise NotImplementedError

        if not os.path.exists(wav_res_path):
            continue
        params.append((wav_res_path, text_ref))
    fout = open(res_path, "w")

    n_higher_than_50 = 0
    wers_below_50 = []
    for wav_res_path, text_ref in tqdm(params):
        transcription = model.transcribe(wav_res_path, language="en", task="transcribe", temperature=0.0, fp16=False)['text']

        raw_truth, raw_hypo, wer, subs, dele, inse = process_one(transcription, text_ref)
        fout.write(f"{wav_res_path}\t{wer}\t{raw_truth}\t{raw_hypo}\t{inse}\t{dele}\t{subs}\n")
        fout.flush()

run_asr(wav_res_text_path, res_path)
