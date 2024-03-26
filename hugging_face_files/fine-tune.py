import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2Model, Trainer, TrainingArguments


def load_and_tokenize_data():
    dataset = load_dataset("text", data_files={"train": "../data/output_train.txt",
                                               "validation": "../data/output_val.txt"})
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=1, remove_columns=["text"])

    return tokenized_datasets["train"], tokenized_datasets["validation"], tokenizer


# Define the evaluation metric (e.g., perplexity)
def compute_metrics(eval_prediction):
    perplexity = eval_prediction["loss"].exp().item()
    return {"perplexity": perplexity}


def train(model, train_dataset, val_dataset, output_dir, device, epochs=3, batch_size=4):
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_first_step=True,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,  # Pass the tokenizer for evaluation
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    train_dataset, val_dataset, tokenizer = load_and_tokenize_data()

    model = GPT2Model.from_pretrained("gpt2").to(device)

    train(model, train_dataset, val_dataset, "../model", device)
