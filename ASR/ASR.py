import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import os
import pandas as pd
import json
import requests
import whisper
from jiwer import wer, cer

df = pd.read_excel('./MathSpeech.xlsx')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model_base = whisper.load_model('base')
whisper_model_base.to(device)

whisper_model_small = whisper.load_model('small')
whisper_model_small.to(device)


originalNT = df['transcription']

base_result_list = []
small_result_list = []

for i in range(0,1101):
    try: 
        out_fn = f'./MathSpeech/{i+1}.mp3'
        
        result1 = whisper_model_base.transcribe(out_fn, language='en')     
        raw_script1 = result1['text']
        print(f'whisper base result : {raw_script1}')
        base_result_list.append(raw_script1)

        result2 = whisper_model_small.transcribe(out_fn, language='en') 
        raw_script2 = result2['text']
        print(f'whisper small result : {raw_script2}')
        small_result_list.append(raw_script2)


        print(f'original NT text : {originalNT[i]}')
        
        print(f"=================================epoch {i} finish==================================================")
    except:
        print(f"index {i+1} is None.")

df["whisper_base_predSE"] = base_result_list
df["whisper_small_predSE"] = small_result_list

df.to_csv('../Experiments/result_ASR.csv', index=False)
df.to_csv('../Ablation_Study/result_ASR.csv', index=False)
