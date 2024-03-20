from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
import evaluate


def load_and_tokenize_data():
    dataset = load_dataset("text", data_files={"train": "../data/output_train.txt",
                                               "test": "../data/output_test.txt",
                                               "validation": "../data/output_val.txt"})
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    val_dataset = tokenized_datasets["validation"].shuffle(seed=42)
    test_dataset = tokenized_datasets["test"].shuffle(seed=42)

    return train_dataset, val_dataset, test_dataset, tokenizer


def create_model():
    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
    return model


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train_model(trainer):
    trainer.train()


def main(device):
    train_dataset, val_dataset, test_dataset, tokenizer = load_and_tokenize_data()
    model = create_model()
    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="epoch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        decvice=device
    )

    train_model(trainer)


if __name__ == "__main__":
    decvice = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
    main(decvice)
