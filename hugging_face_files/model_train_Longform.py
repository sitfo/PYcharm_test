import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import GPT2TokenizerFast, AutoModelForCausalLM, AdamW


class StoryDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        tokenized_text = self.tokenized_texts[idx]
        input_ids = tokenized_text['input_ids']
        attention_mask = tokenized_text['attention_mask']
        return {'input_ids': torch.tensor(input_ids), 'attention_mask': torch.tensor(attention_mask)}


def load_and_tokenize_data():
    dataset = load_dataset("text", data_files={"train": "../data/output_train.txt",
                                               "validation": "../data/output_val.txt"})
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Add padding token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add padding token to tokenizer

    # Preprocess text data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512, return_tensors='pt')

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=1, remove_columns=["text"])
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    val_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(1000))

    train_texts = []
    for train_example in train_dataset:
        train_texts.append({'input_ids': train_example['input_ids'], 'attention_mask': train_example['attention_mask']})

    val_texts = []
    for val_example in val_dataset:
        val_texts.append({'input_ids': val_example['input_ids'], 'attention_mask': val_example['attention_mask']})

    return train_texts, val_texts, tokenizer


def create_model():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return model


def train(model, train_loader, val_loader, device, epochs=3, model_save_dir="../model"):
    os.makedirs(model_save_dir, exist_ok=True)

    writer = SummaryWriter()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    best_val_loss = float('inf')
    best_model_path = os.path.join(model_save_dir, "best_model.pt")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, batch in progress_bar:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)

            optimizer.zero_grad()

            outputs = model(inputs, attention_mask=masks)
            logits = outputs.logits

            loss = criterion(logits.view(-1, logits.shape[-1]), inputs.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                writer.add_scalar('training loss', running_loss / 100, epoch * len(train_loader) + i)
                running_loss = 0.0

            progress_bar.set_description(f"Epoch {epoch + 1}/{epochs}")
            progress_bar.set_postfix(loss=running_loss)

        epoch_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                inputs = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)

                outputs = model(inputs, attention_mask=masks)
                logits = outputs.logits

                loss = criterion(logits.view(-1, logits.shape[-1]), inputs.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('validation loss', avg_val_loss, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

    # Close tensorboard writer
    writer.close()


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
    print(device)
    train_texts, val_texts, tokenizer = load_and_tokenize_data()

    train_dataset = StoryDataset(train_texts)
    val_dataset = StoryDataset(val_texts)

    model = create_model().to(device)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    train(model, train_loader, val_loader, device)
