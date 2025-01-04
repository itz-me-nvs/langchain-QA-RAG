from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
# from langchain.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Step 1: Configure the embedding model
embed_model_name = "sentence-transformers/all-mpnet-base-v2"
embed_model = HuggingFaceEmbeddings(model_name=embed_model_name)

# Step 2: Example documents for testing
documents = [
    {"content": "LangChain enables developers to create applications using LLMs."},
    {"content": "FAISS is a library for efficient similarity search and clustering."},
]

# Step 3: Create a vector store using FAISS
texts = [doc["content"] for doc in documents]
faiss_store = FAISS.from_texts(texts, embed_model)

# Step 4: Load LLaMA-2 model
llama_model_name = "meta-llama/Llama-2-7b-chat-hf"
light_weight_model_name="distilgpt2"; # Lightweight causal language model

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
model = AutoModelForCausalLM.from_pretrained(
    llama_model_name,
    torch_dtype=torch.float16,  # Use float16 for better performance on GPU
    device_map="auto",
)

# using lightweight model for fast performance and reduce power consumption
# model = AutoModelForCausalLM.from_pretrained(light_weight_model_name,
#     torch_dtype=torch.float16,  # Use float16 for better performance if using GPU
#     device_map="auto",  # Automatically map model to available devices
# )

# Set up a HuggingFace pipeline
pipeline_llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7,
    top_p=0.95,
)

# Step 5: Create the LLM
llm = HuggingFacePipeline(pipeline=pipeline_llm)

# Step 6: Build the Retrieval QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=faiss_store.as_retriever(),
    return_source_documents=True,
)

# print("QA Chain:", qa_chain)

# Step 7: Ask a question
# query = "What is LangChain?"
query = "What is 2 + 2 ?"
# response = qa_chain({"query": query})

response=qa_chain.invoke({"query": query})

answer = response["result"]
# response = qa_chain({"query": query})

print("Answer:", answer)
