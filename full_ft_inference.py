import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse
import os

def normalize_text(model, tokenizer, text, max_length=512):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=max_length)
    normalized_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return normalized_text

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory of the saved model and tokenizer checkpoints')
args = parser.parse_args()

# Load the tokenizer and model from the checkpoint directory
checkpoint_dir = args.checkpoint_dir

# Check if the directory exists
if not os.path.exists(checkpoint_dir):
    raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir)

# Move model to appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Example inference with increased max_length
text = "08-09-2024"
normalized_text = normalize_text(model, tokenizer, text, max_length=512)
print(f"Normalized Text: {normalized_text}")

# Debugging: Print tokenized input and output ids
input_ids = tokenizer(text, return_tensors="pt").input_ids
print(f"Tokenized Input IDs: {input_ids}")

with torch.no_grad():
    outputs = model.generate(input_ids.to(next(model.parameters()).device), max_length=512)
output_ids = outputs[0]
print(f"Output IDs: {output_ids}")

# Decode the output IDs
decoded_output = tokenizer.decode(output_ids, skip_special_tokens=False)
print(f"Decoded Output: {decoded_output}")