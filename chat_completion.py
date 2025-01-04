from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer=GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id)

sequence = "How to say 'Thank you' in Spanish?"

inputs = tokenizer.encode(sequence, return_tensors='pt')

# generating text
outputs = model.generate(inputs, max_length=200, do_sample=False, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

# decoding text
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# printing output
print(text)