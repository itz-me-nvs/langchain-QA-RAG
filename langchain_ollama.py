from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st  # type: ignore


# Initialize session state for input field
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries."),
        ("user", "Question: {question}")
    ]
)

# Streamlit framework setup
st.title("LangChain with Ollama Model")
input_text = st.text_input("Enter your question", value=st.session_state.input_text)

# Initialize the Ollama LLM with the desired model
llm = Ollama(model="llama3.2:latest")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# Submit button
if st.button("Submit"):
    if input_text:
        response = chain.invoke({"question": input_text})
        st.write(response)
