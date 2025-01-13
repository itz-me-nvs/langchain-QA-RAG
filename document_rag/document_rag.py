from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.llms import OpenAI

# ollama llm model (local model)
from langchain_community.llms import Ollama

from langchain_core.messages import AIMessage

from langchain_core.output_parsers import StrOutputParser

# Load environment variables from the .env file
load_dotenv()

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

for doc in docs:
    print("metaData:",doc.metadata)

# Check the length of the document
len(docs)

# Check the content of the first document
# print(docs[0].page_content
# )
# Persist the database to use it later
# vectordb.persist()


# question = "what did they say about regression in the third lecture?"

# docs = vectordb.similarity_search(question,k=5)


# # Print the metadata of the similarity search result
# for doc in docs:
#     print(doc.metadata)

# print(docs[4].page_content)


texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]

smalldb = Chroma.from_texts(
    texts=texts,
    embedding=embedding
)

question = "Tell me about all-white mushrooms with large fruiting bodies"

text_mmr = smalldb.max_marginal_relevance_search(
    query=question,
    k=2,
    fetch_k=3
)

print(text_mmr[0].page_content)



# compare similarity & MMR(Maximum Marginal Relevance) search

question = "what did they say about matlab?"
docs_ss=vectordb.similarity_search(question, k=3)
print("result 1:", docs_ss[0].page_content[:100])
print("result 2:", docs_ss[1].page_content[:100])


docs_mmr=vectordb.max_marginal_relevance_search(question, k=3)
print("result 1:", docs_mmr[0].page_content[:100])
print("result 2:", docs_mmr[1].page_content[:100])


# Self Query - Self Query is an important tool when we want to infer metadata from the query itself. We can use SelfQueryRetriever, which uses an LLM to extract
# The query string to use for vector search
# A metadata filter to pass in as well

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The source path of the document",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page number within the document.",
        type="integer",
    ),
]


document_content_description = "machine learning lecture documents"


llm = Ollama(model="llama3:latest")
# llm = OpenAI(temperature=0)

retriever=SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True
)

parser = StrOutputParser()

chain = llm | retriever | parser

# question = "what did they say about regression in the third lecture?"
question = "What is discussed in the first document?"

try:
    docs = chain.invoke(question)
except StrOutputParser.ParseError as e:
    print(e)
    print("LLM output:", e.llm_output)
    raise

print("docs", len(docs))

for index, doc in enumerate(docs):
    print(f"document : ${index}",doc.page_content[:100])