import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast, AutoModelForCausalLM
from tqdm import tqdm
import os

class StoryDataset(Dataset):
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        tokenized_text = self.tokenized_texts[idx]
        return tokenized_text['input_ids'], tokenized_text['attention_mask']

def load_and_tokenize_data():
    dataset = load_dataset("text", data_files={"train": "../data/output_train.txt",
                                               "validation": "../data/output_val.txt"})
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512, return_tensors='pt')

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=1, remove_columns=["text"])
    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    val_dataset = tokenized_datasets["validation"].shuffle(seed=42)

    train_texts = [{'input_ids': example['input_ids'], 'attention_mask': example['attention_mask']} for example in train_dataset]
    val_texts = [{'input_ids': example['input_ids'], 'attention_mask': example['attention_mask']} for example in val_dataset]

    return train_texts, val_texts, tokenizer

def create_model():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return model

def train(model, train_loader, val_loader, device, epochs=3, model_save_dir="../model"):
    os.makedirs(model_save_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    best_val_loss = float('inf')
    best_model_path = os.path.join(model_save_dir, "best_model.pt")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, batch in progress_bar:
            inputs, masks = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs, attention_mask=masks)
            logits = outputs.logits

            loss = criterion(logits.view(-1, logits.shape[-1]), inputs.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                running_loss = 0.0

            progress_bar.set_description(f"Epoch {epoch + 1}/{epochs}")
            progress_bar.set_postfix(loss=running_loss)

        epoch_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                inputs, masks = batch[0].to(device), batch[1].to(device)

                outputs = model(inputs, attention_mask=masks)
                logits = outputs.logits

                loss = criterion(logits.view(-1, logits.shape[-1]), inputs.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    train_texts, val_texts, tokenizer = load_and_tokenize_data()

    train_dataset = StoryDataset(train_texts)
    val_dataset = StoryDataset(val_texts)

    model = create_model().to(device)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    train(model, train_loader, val_loader, device)
