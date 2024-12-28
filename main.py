# Import necessary packages
import os
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import validators

# Load OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secret("key")

# Streamlit user interface
st.title("Conversational Website Chatbot")
st.write("Enter a valid website to start processing its contents")

# Get URL
url = st.text_input("Enter a website URL: ")

if url:
  if validators.url(url):
    st.success("Valid URL. Processing the content from the website...")
  else:
    st.error("Invalid URL. Please enter a valid website")
