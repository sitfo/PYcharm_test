import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

# Create a SummaryWriter instance
writer = SummaryWriter()


def load_dataset(train_path, test_path, tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=512)

    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=512)

    return train_dataset, test_dataset


def train(model, train_dataset, test_dataset, output_dir, device, data_collator):
    """
    Train the model.

    Args:
        model (GPT2LMHeadModel): The model to be trained.
        train_dataset (TextDataset): The training dataset.
        test_dataset (TextDataset): The testing dataset.
        output_dir (str): The directory where the model will be saved.
        device (torch.device): The device where the model will be trained.
    """

    # Set the batch size and the learning rate
    batch_size = 4
    learning_rate = 1e-5

    # Create data loaders
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               collate_fn=data_collator)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              collate_fn=data_collator)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Set the model to training mode
    model.train()

    # Initialize the best loss to a very high value
    best_loss = float('inf')

    # Train the model
    for epoch in range(3):  # Number of epochs
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch")
        for batch in progress_bar:
            optimizer.zero_grad()

            # Forward pass
            inputs = batch["input_ids"].to(device)
            labels = batch["input_ids"].roll(-1).to(device)
            outputs = model(inputs, labels=labels)
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        total_loss = 0
        progress_bar = tqdm(test_loader, desc="Validation", unit="batch")
        with torch.no_grad():
            for batch in progress_bar:
                inputs = batch["input_ids"].to(device)
                labels = batch["input_ids"].roll(-1).to(device)
                outputs = model(inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                # Update the progress bar
                progress_bar.set_postfix({'validation_loss': loss.item()})

        avg_loss = total_loss / len(test_loader)
        print(f"Validation loss: {avg_loss}")

        # Log the average loss to TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch)

        # If this epoch's loss is lower than the best loss, update the best loss and save the model
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained(output_dir)

        model.train()


if __name__ == "__main__":
    """
    Main function to execute the training process.
    """

    # Initialize the distributed environment using Slurm arguments
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.distributed.init_process_group("nccl", rank=local_rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    # Move model and datacollator to the GPU
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Wrap the model with FullyShardedDataParallel
    model = FSDP(model)
    model = model.to(device)

    # Set up the remaining variables
    train_path = '../data/output_train.txt'
    test_path = '../data/output_val.txt'

    # Load the datasets
    train_dataset, test_dataset = load_dataset(train_path, test_path, tokenizer)

    output_dir = "/rds/projects/l/leemg-jinlongphd/models"
    train(model, train_dataset, test_dataset, output_dir, device, data_collator)

    # After finishing training, don't forget to destroy the process group
    destroy_process_group()
