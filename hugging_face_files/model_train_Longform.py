import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from torch.utils.tensorboard import SummaryWriter

def load_and_tokenize_data():
    dataset = load_dataset("text", data_files={"train": "../data/output_train.txt",
                                               "validation": "../data/output_val.txt"})
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    val_dataset = tokenized_datasets["validation"].shuffle(seed=42)

    return train_dataset, val_dataset, tokenizer

def create_model():
    model = AutoModelForCausalLM.from_pretrained("google-bert/bert-base-cased")
    return model

def compute_loss(model, data):
    inputs = data['input_ids'].to(device)
    outputs = model(inputs, labels=inputs)  # Use the inputs as the labels
    return outputs.loss

def train(model, train_loader, val_loader, device, epochs=3):
    optimizer = AdamW(model.parameters(), lr=1e-5)
    best_loss = float('inf')
    writer = SummaryWriter()  # Create a SummaryWriter object

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = compute_loss(model, batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.empty_cache()  # Free up GPU memory

        model.eval()
        total_loss = 0
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                loss = compute_loss(model, batch)
                total_loss += loss.item()
                if loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(), './model/best_model.pt')
        avg_loss = total_loss / len(val_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)  # Log the average loss to TensorBoard

    writer.close()  # Close the SummaryWriter when you're done with it

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_dataset, val_dataset, tokenizer = load_and_tokenize_data()
    model = create_model().to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    train(model, train_loader, val_loader, device)