from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
from datasets import load_dataset



tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large")

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

data = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')

def train_model():
    model.train()



def generate_text(prompt, max_length=500, tempurature=0.7, top_k=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=tempurature,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )
    generate_text= tokenizer.decode(output[0], skip_special_tokens=True)
    return generate_text

