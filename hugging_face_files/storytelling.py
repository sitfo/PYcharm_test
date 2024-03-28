from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained("/rds/projects/l/leemg-jinlongphd/models")
    return model, tokenizer


def generate_story(model, tokenizer, prompt, max_length=1000, top_k=50):
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    # Create attention mask
    attention_mask = inputs.ne(tokenizer.pad_token_id).float()

    outputs = model.generate(inputs,
                             max_length=max_length,
                             do_sample=True,
                             temperature=0.7,
                             top_k=top_k,
                             attention_mask=attention_mask,
                             pad_token_id=tokenizer.eos_token_id)

    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story

if __name__ == "__main__":
    model_path = "/rds/projects/l/leemg-jinlongphd/models"  # replace with the path to your fine-tuned model
    model, tokenizer = load_model(model_path)

    prompt = "write a story about a man and a dog"  # replace with your story prompt
    story = generate_story(model, tokenizer, prompt)
    print(story)