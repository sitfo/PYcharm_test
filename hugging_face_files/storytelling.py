from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    return model, tokenizer

def generate_story(model, tokenizer, prompt, max_length=1000, top_k=50):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, do_sample=True, temperature=0.7, top_k=top_k)
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story

if __name__ == "__main__":
    model_path = "../model"  # replace with the path to your fine-tuned model
    model, tokenizer = load_model(model_path)

    prompt = "Once upon a time"  # replace with your story prompt
    story = generate_story(model, tokenizer, prompt)
    print(story)