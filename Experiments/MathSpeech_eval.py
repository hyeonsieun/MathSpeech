import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import os
import pandas as pd
import json
import requests

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_corrector = "AAAI2025/MathSpeech_Ablation_Study_LaTeX_translator_T5_small" # All T5 models were trained using the same tokenizer.

tokenizer = T5Tokenizer.from_pretrained(path_corrector)


model_corrector= T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
model_corrector.resize_token_embeddings(len(tokenizer))
model_corrector.to(device)

model_trans = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
model_trans.resize_token_embeddings(len(tokenizer))
model_trans.to(device)

df = pd.read_csv('./result_ASR.csv')

original = df['transcription']
beam_result1 = df['whisper_base_predSE'] # The 1st candidate for ASR, which contains an error
beam_result2 = df['whisper_small_predSE'] # The 2nd candidate for ASR, which contains an error

MAX_LENGTH1 = 540
MAX_LENGTH2 = 275
MAX_LENGTH3 = 275

class MathASR(torch.nn.Module):
    def __init__(self, tokenizer, model_name1, model_name2, device):
        super(MathASR, self).__init__()
        self.tokenizer = tokenizer
        self.model1 = model_name1
        self.model1.to(device)

        self.model2 = model_name2
        self.model2.to(device)

        self.device = device

    def forward(self, input_ids, attention_mask_correct, attention_mask_translate, labels_correct, labels_translate):
        # First T5 model forward pass
        input_ids = input_ids.contiguous()
        attention_mask_correct = attention_mask_correct.contiguous()
        labels_correct = labels_correct.contiguous()
        attention_mask_translate = attention_mask_translate.contiguous()
        labels_translate = labels_translate.contiguous()

        outputs1 = self.model1(input_ids=input_ids, attention_mask=attention_mask_correct, labels=labels_correct)
        loss1 = outputs1.loss
        logits1 = outputs1.logits

        # Generate intermediate output from the first T5 model
        intermediate_ids = torch.argmax(logits1, dim=-1).detach()

        # Second T5 model forward pass
        outputs2 = self.model2(input_ids=intermediate_ids, attention_mask=attention_mask_translate, labels=labels_translate)
        loss2 = outputs2.loss

        # Total loss
        total_loss = 0.3*loss1 + 0.7*loss2

        return total_loss, outputs1.logits, outputs2.logits


model = MathASR(tokenizer = tokenizer, model_name1 = model_corrector, model_name2 = model_trans,  device = device)



model.load_state_dict(torch.load('./MathSpeech_checkpoint.pth')) # Load model weights. You need to write the path where the weights are stored.

model.to(device)

model.eval()

ours_list = []
for i in range(0, len(original)):
    input_text = f"translate ASR to truth: {beam_result1[i]} || {beam_result2[i]}"
    inputs = tokenizer.encode(
        input_text,
        return_tensors='pt',
        max_length=MAX_LENGTH1,
        padding='max_length',
        truncation=True
    ).to('cuda')
    corrected_ids = model.model1.generate(
        inputs,
        max_length=MAX_LENGTH2,
        num_beams=5, # `num_beams=1` indicated temperature sampling.
        early_stopping=True
    )
    corrected_and_postprocess = corrected_ids[0][1:-1]
    #print(f"=================={corrected_and_postprocess}===================")
    corrected_sentence = tokenizer.decode(
        corrected_and_postprocess,
        skip_special_tokens=False
    )
    inputs2 = tokenizer.encode(
        corrected_sentence,
        return_tensors='pt',
        max_length=MAX_LENGTH2,
        padding='max_length',
        truncation=True
    ).to('cuda')
    latex_ids = model.model2.generate(
        inputs2,
        max_length=MAX_LENGTH3,
        num_beams=5, # `num_beams=1` indicated temperature sampling.
        early_stopping=True
    )
    latex_and_postprocess = latex_ids[0][1:-1]
    #print(f"=================={latex_and_postprocess}===================")
    latex_output = tokenizer.decode(
        latex_and_postprocess,
        skip_special_tokens=False
    )
    latex_output = latex_output.replace(" ", "")
    print(latex_output)
    ours_list.append(latex_output)
    print(f"------------------------------------------{i} end------------------------------------------")

df["MathSpeech_LaTeX_result"] = ours_list
df.to_csv('MathSpeech_LaTeX_result.csv', index=False)