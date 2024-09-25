import streamlit as st

# Create the LLM
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.1,
    max_retries=2,
    api_key=st.secrets["GROQ_API_KEY"]

)

# Create the Embedding model
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=st.secrets['HF_API'], model_name="sentence-transformers/all-MiniLM-l6-v2"
)