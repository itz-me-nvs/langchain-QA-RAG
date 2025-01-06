from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Load PDF

loaders = [
PyPDFLoader("../docs/machinelearning-lecture01.pdf"),
PyPDFLoader("../docs/machinelearning-lecture02.pdf"),
PyPDFLoader("../docs/machinelearning-lecture03.pdf"),
]

docs = []

for loader in loaders:
    docs.extend(loader.load())


# Define the Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits=text_splitter.split_documents(docs)
print(splits, len(splits))


# initialize embedding

# embedding = OpenAIEmbeddings()

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

persist_directory = 'docs/chroma/'

# Create the vector store
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())

# similarity search

question = "is there an email i can ask for help"
docs = vectordb.similarity_search(question,k=3)

# Check the length of the document
len(docs)

# Check the content of the first document
print(docs[0].page_content
)
# Persist the database to use it later
# vectordb.persist()


# question = "what did they say about regression in the third lecture?"

# docs = vectordb.similarity_search(question,k=5)


# # Print the metadata of the similarity search result
# for doc in docs:
#     print(doc.metadata)

# print(docs[4].page_content)