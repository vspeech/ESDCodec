import esdcodec

w2v_path = "./w2v-bert-2.0" # your downloaded path
esdcodec_model_path = "./esdcodec_ckpts" # your downloaded path
model_id = "25hz_v1"

esdcodec_model = esdcodec.get_model(model_id, esdcodec_model_path)
esdcodec_inference = esdcodec.Inference(esdcodec_model=esdcodec_model, esdcodec_path=esdcodec_model_path, device="cuda")

# do inference for your wav
import torchaudio
import os
import sys

out_dir = "out_wavs"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# test.lst is your wav list
with open('test.lst', 'r') as fin:
    lines = fin.readlines()

for line in lines:
    line = line.strip()
    print(line)
    na = line.split('/')[-1].split('.')[0]
    out_wav = out_dir + "/" + na + "_generated.wav"
    audio, sr = torchaudio.load(line)
    
    # resample to 24kHz
    audio = torchaudio.functional.resample(audio, sr, 24000)
    audio = audio.reshape(1,1,-1)
    audio = audio.to("cuda")
    semantic_codes, acoustic_codes = esdcodec_inference.encode(audio, n_quantizers=3)

    out_audio = esdcodec_inference.decode(semantic_codes, acoustic_codes)

    # save output audio
    torchaudio.save(out_wav, out_audio.cpu().squeeze(0), 24000, encoding="PCM_S", bits_per_sample=16)
