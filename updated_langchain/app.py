from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.chat_models import ChatOpenAI

from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY']= "123"
os.environ['LANGSMITH_API_KEY']= os.getenv("LANGSMITH_API_KEY")
os.environ['LANGSMITH_TRACING']= os.getenv("LANGSMITH_TRACING")


app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server",
    openapi_url=None
)

# Use Ollama with Qwen 2.5 model
llm = Ollama(model="qwen2.5:0.5b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Write me an poem about {topic} for a 5 years child with 5 words"
)

# Add LangServe route
add_routes(
    app,
    prompt|llm,
    path="/poem"
)

# Optional: Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
