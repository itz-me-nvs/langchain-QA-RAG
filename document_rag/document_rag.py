from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
# from langchain.llms import OpenAI

# ollama llm model (local model)
# from langchain_community.llms import Ollama

# from langchain_core.messages import AIMessage

# from langchain_core.output_parsers import JsonOutputParser
# from langchain.schema import OutputParserException, Document
from langchain_groq import ChatGroq # type: ignore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

groq_api_key = os.getenv("GROQ_API_KEY")

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
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits=text_splitter.split_documents(docs)
print(splits, len(splits))


# HuggingFace Embedding
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

persist_directory = 'docs/chroma/'

# Create the vector store
vectordb = Chroma.from_documents(
    documents=docs,
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
print(docs[0].page_content
)
# Persist the database to use it later
# vectordb.persist()



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


# llm = Ollama(model="llama3:latest", base_url="http://localhost:11434")

# llm = ChatGroq(model_name="llama-3.3-70b-versatile",temperature=0.7)
llm = ChatGroq(model_name="gemma2-9b-it",temperature=0.7)
compressor = LLMChainExtractor.from_llm(llm)


question = "Give me the name only of the instructor specified in the page 0"

# retriever vs compressed retriever

retriever=SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True
)

compression_retriever = ContextualCompressionRetriever(
   base_compressor=compressor,
   base_retriever=vectordb.as_retriever()
)

# result = retriever.get_relevant_documents(question)
# print("retriever", result[0].page_content)

# Contextual Compression - Compression is another approach to improve the quality of retrieved docs. Since passing the full document through the application can lead to more expensive LLM calls and poorer response, it is useful to pull out only the most relevant bits of the retrieved passages.

compressed_docs = compression_retriever.get_relevant_documents(question)
# pretty_print_docs(compressed_docs)

response = compression_retriever.invoke(question)
# print("compression_retriever", response[0].page_content)



# question = "How many years Andrew Ng Worked as a machine learning researcher?"
# # docs = vectordb.similarity_search(question,k=3)
# compressed_docs = compression_retriever.get_relevant_documents(question)

# compressed_docs_result = compression_retriever.invoke(question)
# print("years worked",compressed_docs_result[0].page_content)



# Retrievel QA Chain.

# build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template) # run chain

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=vectordb.as_retriever(),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
# )


# question = "What are major topics for this class?"
# result = qa_chain({"query": question})
# print("qa_chain", result["result"]) # The provided text does not explicitly mention the major topics of the class. It discusses the class structure, online resources, and assignments, but does not delve into the specific topics that will be covered. thanks for asking!


qa_chain_mr = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)

# question = "What are major topics for this class?"
# question = "Summurize the lecture03 document in two sentences?"
question = "What is the total duration specified in the lecture03 document?"
result = qa_chain_mr({"query": question})
print("qa_chain_mr", result["result"])