from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

from langchain.chains.query_constructor.base import AttributeInfo
from langchain_groq import ChatGroq # type: ignore
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
import os

groq_api_key = "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"

# os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Load environment variables from the .env file
load_dotenv()


def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


# Load PDF

loaders = [
PyPDFLoader("../docs/machinelearning-lecture01.pdf"),
PyPDFLoader("../docs/machinelearning-lecture01.pdf"),
PyPDFLoader("../docs/machinelearning-lecture02.pdf"),
PyPDFLoader("../docs/machinelearning-lecture03.pdf"),
]

docs = []

for loader in loaders:
  docs.extend(loader.load())


# Define the Text Splitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1500,
#     chunk_overlap = 150
# )

# splits=text_splitter.split_documents(docs)
# print(splits, len(splits))


# HuggingFace Embedding
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

persist_directory = 'docs/chroma/'

# Create the vector store
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory=persist_directory
)

# Self Query - Self Query is an important tool when we want to infer metadata from the query itself. We can use SelfQueryRetriever, which uses an LLM to extract
# The query string to use for vector search
# A metadata filter to pass in as well

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The source of the document, such as a file name or URL",
        type="string"
    ),
    AttributeInfo(
        name="page",
        description="The page number where the content is located",
        type="integer"
    ),
]


document_content_description = "Machine Learning Lecture Notes"

llm = ChatGroq(model_name="gemma2-9b-it",temperature=0.7, api_key=groq_api_key)
compressor = LLMChainExtractor.from_llm(llm)


# compression_retriever = ContextualCompressionRetriever(
#    base_compressor=compressor,
#    base_retriever=vectordb.as_retriever()
# )

# Retrievel QA Chain.

qa_chain_mr = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)

# question = "Summurize the content in the lecture 03?"
question = "How many times the 'cs229-qa@cs.stanford.edu12' specified in the document?"
result = qa_chain_mr({"query": question})

print("<---------------------------------------------------------------------------------->")
print("Final Result:", result["result"])