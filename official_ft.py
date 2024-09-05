import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import argparse
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
import torch
import evaluate
from transformers import default_data_collator

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='Train or continue training T5 model.')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='Path to save or load model checkpoints')
parser.add_argument('--train_from_scratch', action='store_true', help='Flag to train model from scratch')
args = parser.parse_args()

# Load the model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')

if args.train_from_scratch:
    print("Training from scratch")
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
else:
    checkpoint = args.checkpoint_dir if args.checkpoint_dir else "/work/tc062/tc062/haanh/full_ft/checkpoints"
    if not os.path.exists(checkpoint) or not os.path.isfile(os.path.join(checkpoint, "config.json")):
        print(f"Checkpoint directory {checkpoint} does not contain a valid model. Training from scratch instead.")
        model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
    else:
        print(f"Training from checkpoint: {checkpoint}")
        model = T5ForConditionalGeneration.from_pretrained(checkpoint)

# Load the dataset
df = pd.read_csv('/work/tc062/tc062/haanh/full_ft/no_long_token.csv', encoding="utf-8", delimiter=',')
dataset = Dataset.from_pandas(df[['sentences', 'normalizations']])
train_val_dataset, test_dataset = dataset.train_test_split(test_size=0.1).values()
train_dataset, val_dataset = train_val_dataset.train_test_split(test_size=0.1111).values()

# Select 1000 examples for validation
val_dataset = val_dataset.select(range(1000))


# Define custom dataset class
class MyDataset(TorchDataset):
    """
    Custom Dataset class to handle tokenization and formatting for FLAN-T5 model.

    Args:
        dataset (Dataset): The dataset containing input sentences and target normalizations.
        tokenizer (T5Tokenizer): The tokenizer for encoding inputs and targets.
        eval (bool): Flag to indicate whether the dataset is for evaluation (shorter max lengths).
    """

    def __init__(self, dataset, tokenizer, eval=False):
        self.examples = dataset
        self.tokenizer = tokenizer
        self.in_max_len = 128 if eval else 256
        self.out_max_len = 128 if eval else 512

    def __len__(self):
        """
        Returns the total number of examples in the dataset.
        """
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Fetches and tokenizes a single example from the dataset.

        Args:
            idx (int): Index of the example in the dataset.

        Returns:
            dict: Tokenized inputs and labels for the model.
        """
        example = self.examples[idx]
        inputs = example['sentences']
        targets = example['normalizations']
        inputs = self.tokenizer(inputs, max_length=self.in_max_len, truncation=True, padding='max_length',
                                return_tensors="pt")
        labels = self.tokenizer(targets, max_length=self.out_max_len, truncation=True, padding='max_length',
                                return_tensors="pt")
        labels["input_ids"][labels["input_ids"] == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
        }


# Create DataLoader instances for training, validation, and test datasets
train_dataloader = MyDataset(train_dataset, tokenizer)
val_dataloader = MyDataset(val_dataset, tokenizer, eval=True)
test_dataloader = MyDataset(test_dataset, tokenizer)

# Define evaluation metric
accuracy_metric = evaluate.load("accuracy")


def compute_metrics(pred):
    """
    Compute accuracy metrics for the model's predictions.

    Args:
        pred (EvalPrediction): Predictions and labels from the Trainer.

    Returns:
        dict: Computed accuracy metric.
    """
    print("Starting eval")
    if isinstance(pred.predictions, tuple):
        preds = pred.predictions[0].argmax(axis=-1)
    else:
        preds = pred.predictions.argmax(axis=-1)
    labels = pred.label_ids
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return accuracy_metric.compute(predictions=preds, references=labels)


# Define training arguments for the Trainer
training_args = TrainingArguments(
    output_dir=args.checkpoint_dir,
    evaluation_strategy="steps",
    eval_steps=1000,  # Evaluation is performed every 1000 steps
    save_strategy="steps",
    save_steps=1000,  # Model is saved every 1000 steps
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=2,
    eval_accumulation_steps=16,  # Accumulate evaluation results every 16 steps
    weight_decay=0.01,
    save_total_limit=1,  # Limit the number of saved checkpoints
    num_train_epochs=2,
    load_best_model_at_end=True,  # Load the best model after training
    report_to="none",  # Disable wandb
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=1000,  # Log metrics every 1000 steps
)

# Initialize the Trainer with custom DataLoader and evaluation metric
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader,
    eval_dataset=val_dataloader,
    compute_metrics=compute_metrics,
    data_collator=lambda data: {
        'input_ids': torch.stack([f['input_ids'] for f in data]),
        'attention_mask': torch.stack([f['attention_mask'] for f in data]),
        'labels': torch.stack([f['labels'] for f in data]),
    },
)

# Start training the model from the given checkpoint
trainer.train(checkpoint)

# Start training the model from scratch
# trainer.train()


# Save the trained model and tokenizer to the output directory
output_dir = args.checkpoint_dir if args.checkpoint_dir else "/work/tc062/tc062/haanh/full_ft/final_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Evaluate the model on the validation dataset
metrics = trainer.evaluate(val_dataloader)
accuracy_score = metrics["eval_accuracy"]

# Print the final accuracy score
print(f"My model is done training and the accuracy score is {accuracy_score:.4f}")

