from transformers import pipeline

# Load the Falcon model
qa_pipeline = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")

# Ask a question
question = "Who is the prime minister of India?"
response = qa_pipeline(question, max_length=50, num_return_sequences=1)

# Print the answer
print("Question:", question)
print("Answer:", response[0]['generated_text'])
