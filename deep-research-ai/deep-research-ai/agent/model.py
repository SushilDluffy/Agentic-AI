import os

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

def get_model(model_name,**kwargs):
    model = ChatGoogleGenerativeAI(model=model_name, api_key=GEMINI_KEY,**kwargs)
    return model
