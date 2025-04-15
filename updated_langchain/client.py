import requests
import streamlit as st

# def get_openai_response(input_text):
#     response=requests.post("http://localhost:8000/essay/invoke",
#     json={'input':{'topic':input_text}})

#     return response.json()['output']['content']

def get_ollama_response(input_text):
    response = requests.post(
        "http://localhost:8000/poem/invoke",
        json={'input': {'topic': input_text}}
    )

    # Log the raw response for debugging
    st.write(f"Status code: {response.status_code}")
    # st.write(f"Raw text: {response.text}")

    # Safe parsing
    try:
        return response.json()['output']
    except requests.exceptions.JSONDecodeError as e:
        print('result:', response.text)
        st.error("Response is not JSON. Check the backend API.")
        return None

    ## streamlit framework

st.title('Langchain Demo With LLAMA2 API')
# input_text=st.text_input("Write an essay on")
input_text1=st.text_input("Write a poem on")

# if input_text:
#     st.write(get_openai_response(input_text))

if input_text1:
    st.write(get_ollama_response(input_text1))