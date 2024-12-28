# Install necessary packages
import os
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.openai import OpenAI
from langchain.memory import ConversationalBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import vaildators
