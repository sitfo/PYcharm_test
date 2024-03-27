import os
import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch.nn.parallel import DistributedDataParallel


def load_dataset(train_path, test_path, tokenizer):
    """
    Load the training and testing datasets.

    Args:
        train_path (str): The path to the training dataset.
        test_path (str): The path to the testing dataset.
        tokenizer (GPT2Tokenizer): The tokenizer to be used.

    Returns:
        Tuple[TextDataset, TextDataset]: The training and testing datasets.
    """
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=512)

    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=512)

    return train_dataset, test_dataset


def compute_metrics(eval_pred):
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

    # Cross-Entropy Loss
    ce_loss = F.cross_entropy(predictions, labels)

    # Perplexity
    perplexity = torch.exp(ce_loss)

    # KL Divergence
    kl_div_loss = F.kl_div(predictions, labels)

    # Mean Squared Error
    mse_loss = MSELoss()(predictions, labels)

    return {"cross_entropy": ce_loss.item(), "perplexity": perplexity.item(), "kl_divergence": kl_div_loss.item(),
            "mse": mse_loss.item()}


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
        save_steps=10_000,
        save_total_limit=2,
        evaluation_strategy="epoch",  # Add this line to perform evaluation
        save_strategy="epoch",  # Save strategy is set to "epoch" to match the evaluation strategy
        load_best_model_at_end=True,  # Add this line to load the best model at the endΩ
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
        compute_metrics=compute_metrics,
        # Pass the optimizer to the Trainer
        optimizers=(optimizer, None),
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    """
    Main function to execute the training process.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    model = GPT2LMHeadModel.from_pretrained('gpt2-large')

    model = DistributedDataParallel(model)
    model.to(device)

    train_path = '../data/output_train.txt'
    test_path = '../data/output_val.txt'
    train_dataset, test_dataset = load_dataset(train_path, test_path, tokenizer)

    output_dir = "../model"
    train(model, train_dataset, test_dataset, output_dir, device)
