import time
import openai
import pandas as pd

openai.api_type = "<<PUT YOUE API TYPE>>"
openai.api_base = "<<PUT YOUE API BASE>>"
openai.api_version = "<<PUT YOUE API VERSION>>"
openai.api_key = "<<PUT YOUE API KEY>>"

df = pd.read_csv('./result_ASR.csv')

original = df['transcription']
beam_result1 = df['whisper_base_predSE'] # The 1st candidate for ASR, which contains an error
beam_result2 = df['whisper_small_predSE'] # The 2nd candidate for ASR, which contains an error

GPT_result = []
retries = 3 
for i in range(len(original)):
    prompt = [{
      "role": "system",                         # system role
      "content": """
        You are an AI model that corrects Automatic Speech Recognition results and converts them into LaTeX. Let's think step by step. I will give you two candidates from the Automatic Speech Recognition results. Based on these two ASR results, make the necessary corrections and then output the corresponding LaTeX code for the formula. I will show you one example, so refer to it and convert it into LaTeX. You just have to tell me the answer. Don't tell me anything else, just tell me the LaTeX code.
        Input : a plus seven y plus ten z equals z || a plus 7y plus f z equals 0
        Output : $ a + 7y + 10z = 0 $
      """
    },
    {
     "role": "user",                            # user role
     "content": f"""Input : {beam_result1[i]} || {beam_result2[i]}
     Output : (Tell me Your answer.)"""
    }]

    for attempt in range(retries):
        try:
            model1 = openai.ChatCompletion.create(
                engine="gpt1",
                messages=prompt,
                temperature=0.4,
                max_tokens=800,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            # Check the structure of the response and make sure it has 'content'.
            if 'choices' in model1 and len(model1['choices']) > 0 and 'message' in model1['choices'][0] and 'content' in model1['choices'][0]['message']:
                answer = model1['choices'][0]['message']['content']
                print(f"{answer}")
                GPT_result.append(answer)
            else:
                print("Warning: No content found in response.")
                GPT_result.append("<ERROR>") # If error occurs, add <ERROR> to the result.
            time.sleep(3)
            break  # If successful, stop iterating.
        except openai.error.InvalidRequestError as e:
            print(f"Error: {e}")
            if attempt < retries - 1: 
                print("Waiting for 10 seconds before retrying...")
                time.sleep(10)
            else:
                print("Max retries reached. Moving on to the next item.")
                GPT_result.append("<ERROR>")  # If error occurs, add <ERROR> to the result.

df["GPT3_5_LaTeX_result"] = GPT_result
df.to_csv('GPT3_5_LaTeX_result.csv', index=False)