import streamlit as st

# Create the LLM
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.0,
    max_retries=2,
    api_key=st.secrets["GROQ_API_KEY"]

)

# Create the Embedding model
from sentence_transformers import SentenceTransformer

embeddings = SentenceTransformer('all-MiniLM-L6-v2')

# Create a wrapper class to make it compatible with LangChain
from langchain.embeddings.base import Embeddings

class LocalHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()

# Use the wrapper class
embeddings = LocalHuggingFaceEmbeddings('all-MiniLM-L6-v2')