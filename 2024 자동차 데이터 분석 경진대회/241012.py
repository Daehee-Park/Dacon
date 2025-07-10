import pandas as pd
from tqdm import tqdm

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Prepare the system prompt
system_prompt = (
"You are an brilliant assistant that classifies automobile-related data. Based on the 'title', 'notes' provided for each sample, output a single 0 or 1 for each of the 40 samples. Do not include any explanations or additional text."
)

# Prepare the user prompt with the test data
user_prompts = ["Classify the following 40 samples:\n```samples"]
for idx, row in test.iterrows():
    title = row['title']
    notes = row['notes']
    if pd.notna(notes):
        entry = f"title: {title}\nnotes: {notes}"
    else:
        entry = f"title: {title}"
    user_prompts.append(entry)

# Combine the entries into the user prompt, separated by a delimiter
user_prompt = "\n--\n".join(user_prompts)
user_prompt+="\n--\n```\nRemember, you must print out exactly **40 rows**, each containing only a 0 or 1, corresponding to your classification of each sample."

# Create the submission DataFrames
submission = pd.DataFrame({
    'system': [system_prompt],
    'user': [user_prompt]
})

# Save the submission file
submission.to_csv("Sub/submission_241012_4.csv", index=False)