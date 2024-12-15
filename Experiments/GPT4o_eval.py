import time
import pandas as pd
import os
import json
import requests

API_KEY = "<<PUT YOUE API KEY>>"
headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}
ENDPOINT = "<<PUT YOUE END POINT>>"

df = pd.read_csv('./result_ASR.csv')

original = df['transcription']
beam_result1 = df['whisper_base_predSE'] # The 1st candidate for ASR, which contains an error
beam_result2 = df['whisper_small_predSE'] # The 2nd candidate for ASR, which contains an error

GPT_result = []
retries = 3 


for i in range(len(original)):
  print(f"Input : {beam_result1[i]} || {beam_result2[i]}")
  payload = {
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": """
          You are an AI model that corrects Automatic Speech Recognition results and converts them into LaTeX. Let's think step by step. I will give you two candidates from the Automatic Speech Recognition results. Based on these two ASR results, make the necessary corrections and then output the corresponding LaTeX code for the formula. I will show you one example, so refer to it and convert it into LaTeX. You just have to tell me the answer. Don't tell me anything else, just tell me the LaTeX code.
          Input : a plus seven y plus ten z equals z || a plus 7y plus f z equals 0
          Output : $ a + 7y + 10z = 0 $
          """
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": f"""Input : {beam_result1[i]} || {beam_result2[i]}
                  Output : (Tell me Your answer.)"""
        }
      ]
    },
  ],
  "temperature": 0.4,
  "top_p": 0.95,
  "frequency_penalty": 0,
  "presence_penalty": 0,
  }
  for attempt in range(retries):
      try:
          response = requests.post(ENDPOINT, headers=headers, json=payload)
          response.raise_for_status()  # Verify that there were no problems with the HTTP request
          response_json = response.json()
          # Check if 'choices', 'message', and 'content' keys exist in the response JSON.
          if 'choices' in response_json and response_json['choices']:
              choice = response_json['choices'][0] 
              if 'message' in choice and 'content' in choice['message']:
                  content = choice['message']['content']
              else:
                  content = "<ERROR>"
          else:
              content = "<ERROR>"
          print(f"content : {content}")
          GPT_result.append(content)
          time.sleep(3)
          print(f"===========================epoch {i}===============================")
          break  # If successful, stop iterating.
      except requests.RequestException as e:
          print(f"Error: {e}")
          if attempt < retries - 1: 
              print("Waiting for 20 seconds before retrying...")
              time.sleep(20)
          else:
              print("Max retries reached. Moving on to the next item.")
              GPT_result.append("<ERROR>") 


df["GPT4o_LaTeX_result"] = GPT_result
df.to_csv('GPT4o_LaTeX_result.csv', index=False)
