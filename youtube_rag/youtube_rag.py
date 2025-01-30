from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def extract_text_from_youtube_url(url):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    docs = loader.load()
    return docs

def summarize_text_groq(text, model_name="gemma2-9b-it", temperature=0.7):
    if not text or len(text.strip()) == 0:
        return "Error: No text provided"

    llm = ChatGroq(model_name=model_name, temperature=temperature)

      # Prepare messages
    messages = [
        (
            "system",
            "Summarize the text below into a clear and concise list. Break down the main points into bullet points for easy understanding."
        ),
        ("human", "Here is the text:\n\n" + text),
    ]

    # Generate Summary
    summary = llm.invoke(messages)
    return summary.content


# Main Function

if __name__ == "__main__":
    youtube_url = "https://youtu.be/K6G7E6vCDw0?si=q31j0i6ZtdsK6Nzz"
    extracted_text = extract_text_from_youtube_url(youtube_url)
    print("Extracted Text:\n", extracted_text[0].page_content[:20])  # Print a snippet of the text

    text = extracted_text[0].page_content[:200]


    summary = summarize_text_groq(text)
    print("\nSummary:\n", summary)


