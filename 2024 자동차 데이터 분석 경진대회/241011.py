import pandas as pd
from tqdm import tqdm

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Function to prepare few-shot examples from the training data with CoT reasoning
def prepare_examples(train_data, num_examples=4):
    examples = []
    # Get equal number of positive and negative examples
    pos_samples = train_data[train_data['target'] == 1].sample(num_examples // 2)
    neg_samples = train_data[train_data['target'] == 0].sample(num_examples // 2)
    sampled_examples = pd.concat([pos_samples, neg_samples])
    # Translate and format examples with reasoning
    for _, row in sampled_examples.iterrows():
        title = row['title']
        notes = row['notes']
        label = row['target']
        # Provide reasoning focusing on automotive keywords
        reasoning = (
            f"I examine the title and notes to determine if the content is related to automotive topics. "
            f"The terms '{title}' and '{notes}' {'include' if label == 1 else 'do not include'} automotive-related keywords. "
            f"Therefore, the label is {label}."
        )
        example = f"Title: {title}\nNotes: {notes}\nReasoning: {reasoning}\n"
        examples.append(example)
    return "\n---\n".join(examples)

# Prepare the system prompt with CoT guidance and few-shot examples
examples_text = prepare_examples(train, num_examples=3)
system_prompt = (
    "You are an expert in classifying data entries as automotive-related (1) or not automotive-related (0). "
    "For each entry, think through the reasoning step by step to determine if it's related to automotive topics, focusing on specific automotive-related keywords. "
    "However, do not share your reasoning in the final answer. "
    "Only output the final answer as '0' or '1' without any additional text. "
    "Here are some examples:\n\n"
    f"{examples_text}\n"
    "---\n"
    "Finally, your response should be a series of '0's and '1's, each on a new line, like:\n"
    "1\n1\n0\n\n"
    f"There are **{test.shape[0]}** entries to classify\n"
    f"Ensure your final response contains exactly **{test.shape[0]}** lines, each with either '0' or '1', corresponding to each entry.\n"
    "Do not include any extra text or explanations.\n"
    "Now, classify the following entries:\n\n"
)

# Prepare the user prompt
user_prompts = []
for idx, row in tqdm(test.iterrows(), total=test.shape[0]):
    title = row['title']
    notes = row['notes']
    # Format the entry
    entry = f"Title: {title}\nNotes: {notes}\n"
    user_prompts.append(entry)

user_prompt = "\n---\n".join(user_prompts)

# Create the submission DataFrame
submission = pd.DataFrame({
    'system': [system_prompt],
    'user': [user_prompt]
})

# Save the submission file
submission.to_csv("Sub/submission_241011.csv", index=False)