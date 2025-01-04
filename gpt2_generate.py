from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)

sequence = "What is Javascript"

inputs = tokenizer.encode(sequence, return_tensors='pt')

# generating text
outputs = model.generate(inputs, max_length=100, do_sample=False, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

# decoding text
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# printing output
print(text)