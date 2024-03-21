import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def load_and_tokenize_data():
    dataset = load_dataset("text", data_files={"train": "../data/output_train.txt",
                                               "validation": "../data/output_val.txt"})
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"][:1000].shuffle(seed=42)
    val_dataset = tokenized_datasets["validation"][:1000].shuffle(seed=42)

    return train_dataset, val_dataset, tokenizer


def create_model():
    model = AutoModelForCausalLM.from_pretrained("google-bert/bert-base-cased")
    return model


def train(model, train_loader, val_loader, device, epochs=3):
    optimizer = AdamW(model.parameters(), lr=1e-5)
    best_loss = float('inf')
    writer = SummaryWriter()  # Create a SummaryWriter object

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            reconstructed_batch = model(batch)
            loss = F.mse_loss(reconstructed_batch, batch)  # Reconstruction loss for unsupervised learning
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.empty_cache()  # Free up GPU memory

        model.eval()
        total_loss = 0
        total_seq_acc = 0
        total_token_acc = 0
        total_tokens = 0
        for batch in val_loader:
            batch = batch.to(device)
            with torch.no_grad():
                reconstructed_batch = model(batch)
                loss = F.mse_loss(reconstructed_batch, batch)  # Reconstruction loss for unsupervised learning
                total_loss += loss.item()
                if loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(), './model/best_model.pt')

                # Calculate validation sequence accuracy
                seq_acc = (reconstructed_batch.argmax(dim=-1) == batch).all(dim=-1).float().mean()
                total_seq_acc += seq_acc.item()

                # Calculate validation token accuracy
                token_acc = (reconstructed_batch.argmax(dim=-1) == batch).float().mean()
                total_token_acc += token_acc.item()
                total_tokens += batch.numel()

        avg_loss = total_loss / len(val_loader)
        avg_seq_acc = total_seq_acc / len(val_loader)
        avg_token_acc = total_token_acc / total_tokens

        writer.add_scalar('Loss/train', avg_loss, epoch)  # Log the average loss to TensorBoard
        writer.add_scalar('Accuracy/sequence', avg_seq_acc, epoch)  # Log the sequence accuracy to TensorBoard
        writer.add_scalar('Accuracy/token', avg_token_acc, epoch)  # Log the token accuracy to TensorBoard

    writer.close()  # Close the SummaryWriter when you're done with it


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_dataset, val_dataset, tokenizer = load_and_tokenize_data()
    model = create_model().to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    train(model, train_loader, val_loader, device)
