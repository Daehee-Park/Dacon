import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

request = pd.read_csv("Sub/submission_241012_4.csv")

def response_text(system, user):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.4
    )
    return completion.choices[0].message.content

result = ""
for i in tqdm(range(len(request))):
    result += response_text(request['system'][i], request['user'][i]) + "\n"

with open("result_241012_4.txt", "w") as f:
    f.write(result)