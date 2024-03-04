from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
import evaluate

def load_and_tokenize_data():
    dataset = load_dataset("akoksal/LongForm")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["input"], examples["output"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

    return train_dataset, eval_dataset, tokenizer

def create_model():
    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
    return model

def create_trainer(model, train_dataset, eval_dataset, tokenizer):
    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="epoch",
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    return trainer

def train_model(trainer):
    trainer.train()

def main():
    train_dataset, eval_dataset, tokenizer = load_and_tokenize_data()
    model = create_model()
    trainer = create_trainer(model, train_dataset, eval_dataset, tokenizer)
    train_model(trainer)

if __name__ == "__main__":
    main()