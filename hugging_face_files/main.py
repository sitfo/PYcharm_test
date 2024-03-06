from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForCausalLM, AdamW, get_scheduler
from datasets import load_dataset
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
import tqdm

raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

checkpoint = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, collate_fn=data_collator, batch_size=32
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=32
)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

train_dl, eval_dl, model, optimizer = Accelerator().prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

progress_bar = tqdm.tqdm(range(num_training_steps))

torch_device = "mps" if torch.cuda.is_available() else "cpu"
model.to(torch_device)
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(torch_device) for k, v in batch.items()}
        outputs = model(**batch)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(outputs.logits.view(-1, outputs.logits.shape[-1]), batch["input_ids"].view(-1))
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(torch_device) for k, v in batch.items()}
    outputs = model(**batch)
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(outputs.logits.view(-1, outputs.logits.shape[-1]), batch["input_ids"].view(-1))
    print(loss)

