import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch.nn.parallel import DistributedDataParallel
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


def cross_entropy_loss(predictions, labels):
    return F.cross_entropy(predictions, labels)


def perplexity_loss(predictions, labels):
    ce_loss = cross_entropy_loss(predictions, labels)
    return torch.exp(ce_loss)


def kl_divergence_loss(predictions, labels):
    return F.kl_div(predictions, labels)


def mse_loss(predictions, labels):
    return MSELoss()(predictions, labels)


def compute_metrics(eval_pred, global_step):
    """
    Compute the metrics for evaluation.

    Args:
        eval_pred (Tuple[Tensor, Tensor]): The predictions and labels.

    Returns:
        dict: The computed metrics.
    """
    predictions, labels = eval_pred
    predictions = predictions[0]  # get the logits
    labels = labels[0]  # get the true labels

    # Compute each loss
    ce_loss = cross_entropy_loss(predictions, labels)
    writer.add_scalar('Loss/cross_entropy', ce_loss, global_step)

    # perplexity = perplexity_loss(predictions, labels)
    # writer.add_scalar('Loss/perplexity', perplexity, global_step)
    #
    # kl_div_loss = kl_divergence_loss(predictions, labels)
    # writer.add_scalar('Loss/kl_divergence', kl_div_loss, global_step)
    #
    # mse = mse_loss(predictions, labels)
    # writer.add_scalar('Loss/mse', mse, global_step)

    return {"cross_entropy": ce_loss.item()}


def train(model, train_dataset, test_dataset, output_dir, device):
    """
    Train the model.

    Args:
        model (GPT2LMHeadModel): The model to be trained.
        train_dataset (TextDataset): The training dataset.
        test_dataset (TextDataset): The testing dataset.
        output_dir (str): The directory where the model will be saved.
        device (torch.device): The device where the model will be trained.
    """

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        fp16=True,
        save_steps=10_000,
        save_total_limit=2,
        evaluation_strategy="epoch",  # Add this line to perform evaluation
        save_strategy="epoch",  # Save strategy is set to "epoch" to match the evaluation strategy
        load_best_model_at_end=True,  # Add this line to load the best model at the end
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, trainer.state.global_step),
        # Pass the optimizer to the Trainer
        optimizers=(optimizer, None),
    )

    trainer.train()
    trainer.save_model(output_dir)


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
    model.gradient_checkpointing_enable()
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Wrap the model with FullyShardedDataParallel
    model = FSDP(model)
    model = model.to(device)

    # Set up the remaining variables
    train_path = '/rds/projects/l/leemg-jinlongphd/data/output_train.txt'
    test_path = '/rds/projects/l/leemg-jinlongphd/data/output_val.txt'

    # Load the datasets
    train_dataset, test_dataset = load_dataset(train_path, test_path, tokenizer)

    output_dir = "/rds/projects/l/leemg-jinlongphd/models"
    train(model, train_dataset, test_dataset, output_dir, device)

    # After finishing training, don't forget to destroy the process group
    destroy_process_group()
