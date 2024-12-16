import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer 
import re
import os
import pandas as pd
import json
import requests

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_T5_small = "AAAI2025/MathSpeech_Ablation_Study_1stage_fine-tuned_with_errors_T5_small"

tokenizer_T5_small = T5Tokenizer.from_pretrained(path_T5_small)

model_T5_small = T5ForConditionalGeneration.from_pretrained(path_T5_small)

model_T5_small.to(device)
model_T5_small.eval()


def inference(text, model, tokenizer):
    input_text = f"translate text to LaTeX: {text}"
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

def postprocessing_T5(GT_raw):
    start_index = GT_raw.find("<pad>") + len("<pad>")
    end_index = GT_raw.find("</s>")
    GT_result = GT_raw[start_index:end_index]
    return GT_result


df = pd.read_csv('./result_ASR.csv')

original = df['transcription']
beam_result1 = df['whisper_base_predSE'] # The 1st candidate for ASR, which contains an error
beam_result2 = df['whisper_small_predSE'] # The 2nd candidate for ASR, which contains an error

for i in range(len(original)):
    result_list = []
    input = f"{beam_result1[i]} || {beam_result2[i]}"
    raw_T5_small = inference(input, model_T5_small, tokenizer_T5_small)
    result_T5_small = postprocessing_T5(raw_T5_small)
    result_T5_small = result_T5_small.replace(" ", "")
    result_list.append(result_T5_small)
    print(f"result : {result_T5_small}")

df["MathSpeech_1stage_finetuned_with_error_T5small_result"] = result_list
df.to_csv('MathSpeech_1stage_finetuned_with_error_T5small_result.csv', index=False)