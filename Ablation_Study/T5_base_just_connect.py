import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer 
import re
import os
import pandas as pd
import json
import requests

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_T5_base_1 = "AAAI2025/MathSpeech_Ablation_Study_Error_corrector_T5_base"
tokenizer_T5_base_l = T5Tokenizer.from_pretrained(path_T5_base_1)
model_T5_base_1 = T5ForConditionalGeneration.from_pretrained(path_T5_base_1)
model_T5_base_1.to(device)
model_T5_base_1.eval()

path_T5_base_2 = "AAAI2025/MathSpeech_Ablation_Study_LaTeX_translator_T5_base"
tokenizer_T5_base_2 = T5Tokenizer.from_pretrained(path_T5_base_2)
model_T5_base_2 = T5ForConditionalGeneration.from_pretrained(path_T5_base_2)
model_T5_base_2.to(device)
model_T5_base_2.eval()

def inference_correct(text, model, tokenizer):
    input_text = f"translate ASR to truth: {text}"
    inputs = tokenizer.encode(
        input_text,
        return_tensors='pt',
        max_length=540,
        padding='max_length',
        truncation=True
    ).to('cuda')

    # Get correct sentence ids.
    corrected_ids = model.generate(
        inputs,
        max_length=275,
        num_beams=5, # `num_beams=1` indicated temperature sampling.
        early_stopping=True
    )

    # Decode.
    corrected_sentence = tokenizer.decode(
        corrected_ids[0],
        skip_special_tokens=False
    )
    return corrected_sentence


def inference_trans(text, model, tokenizer):
    input_text = f"{text}"
    inputs = tokenizer.encode(
        input_text,
        return_tensors='pt',
        max_length=275,
        padding='max_length',
        truncation=True
    ).to('cuda')

    # Get correct sentence ids.
    corrected_ids = model.generate(
        inputs,
        max_length=275,
        num_beams=5, # `num_beams=1` indicated temperature sampling.
        early_stopping=True
    )

    # Decode.
    corrected_sentence = tokenizer.decode(
        corrected_ids[0],
        skip_special_tokens=False
    )
    return corrected_sentence

def postprocessing_T5(GT_raw):
    start_index = GT_raw.find("<pad>") + len("<pad>")
    end_index = GT_raw.find("</s>")
    GT_result = GT_raw[start_index:end_index]
    return GT_result


df = pd.read_csv('./result_ASR.csv')

original = df['transcription']
beam_result1 = df['whisper_base_predSE'] # The 1st candidate for ASR, which contains an error
beam_result2 = df['whisper_small_predSE'] # The 2nd candidate for ASR, which contains an error

result_list = []

for i in range(len(original)):
    input = f"{beam_result1[i]} || {beam_result2[i]}"
    T5_correct_raw = inference_correct(input, model_T5_base_1, tokenizer_T5_base_l)
    correct_result = postprocessing_T5(T5_correct_raw)
    print(f"correct_result : {correct_result}")
    raw_T5_base = inference_trans(correct_result, model_T5_base_2, tokenizer_T5_base_2)
    result_T5_base = postprocessing_T5(raw_T5_base)
    result_T5_base = result_T5_base.replace(" ", "")
    result_list.append(result_T5_base)
    print(f"result : {result_T5_base}")

df["MathSpeech_just_connect_T5base_result"] = result_list
df.to_csv('MathSpeech_just_connect_T5base_result.csv', index=False)