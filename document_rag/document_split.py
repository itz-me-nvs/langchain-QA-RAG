from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter
loader=PyPDFLoader("../docs/machinelearning-lecture01.pdf")

# 1. Document Loading

#Load the document by calling loader.load()
pages=loader.load()

# total page length
print(len(pages))

# content of first page
print(pages[0].page_content[0:500])

# meta data about PDF
print(pages[0].metadata)


# 2. Document Splitting

some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""

print(len(some_text)) # 496

# c_splitter = CharacterTextSplitter(
#     chunk_size=200,
#     chunk_overlap=0,
#     separator = '  '
# )

# r_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=450,
#     chunk_overlap=0,
#     separators=["\n\n", "\n", " ", ""]
# )

# # output1=c_splitter.split_text(some_text)
# # print(output1, len(output1))

# output2=r_splitter.split_text(some_text)
# print(output2, len(output2))


text_splitter = CharacterTextSplitter(
 separator="\n",
 chunk_size=1000,
 chunk_overlap=150,
 length_function=len
)

docs=text_splitter.split_documents(pages)

print(len(docs), len(pages))


# markdown header split

markdown_document = """# Title\n\n \
## Chapter 1\n\n \
Hi this is Jim\n\n Hi this is Joe\n\n \
### Section \n\n \
Hi this is Lance \n\n
## Chapter 2\n\n \
Hi this is Molly"""


headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
md_header_splits = markdown_splitter.split_text(markdown_document)
print(md_header_splits)