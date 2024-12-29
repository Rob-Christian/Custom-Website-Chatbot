# Import necessary packages
import os
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import validators

# Load OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["key"]

# Streamlit user interface
st.title("Conversational Website Chatbot")
st.write("Enter a valid website to start processing its contents")

# Get URL
url = st.text_input("Enter a website URL: ")

if url:
  if validators.url(url):
    st.success("Valid URL. Processing the content from the website...")
    try:
      # Get texts from the website
      loaders = UnstructuredURLLoader(urls = [url])
      data = loaders.load()

      # Extract chunks from the website
      text_splitter = CharacterTextSplitter(separator = "/n", chunk_size = 1000, chunk_overlap = 200)
      chunks = text_splitter.split_documents(data)

      # Generate embeddings and create FAISS database
      embeddings = OpenAIEmbeddings()
      vectordb = FAISS.from_documents(documents = chunks, embedding = embeddings)

      # Setup Prompt
      template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

      Chat History:
      {chat_history}
      Follow Up Input: {question}
      Standalone question:"""

      CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template)

      prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

      {context}
      
      Question: {question}
      Helpful Answer:"""

      QA_PROMPT = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

      # Setup LLM
      llm = ChatOpenAI(temperature = 0.2)
      memory = ConversationBufferMemory(
          llm = llm,
          memory_key = 'chat_history',
          output_key = 'answer',
          prompt = QA_PROMPT,
          return_messages = True
      )
    
      # Combine database, LLM, and Memory
      chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectordb.as_retriever(),
        memory = memory
      )

      # Provide options after processing
      st.write("Content processed successfully")
      option = st.radio(
        "What do you like to do next?",
        options = ["Start Chatting", "End"],
        index = 0
      )

      # If start chatting
      if option == "Start Chatting":
        st.write("Chatbot is ready! Start asking questions.")
        user_input = st.text_area("Your question (type exit if you're done asking): ", key = "user_input")

        if user_input:
          if user_input.lower() == "exit":
            st.write("Exiting. Refresh the page to restart")
          else:
            response = chain({"question": user_input})["answer"]
            st.write(f"Chatbot: {response}")

            # Debugging: Display current memory
            st.write("### Debug: Memory Content")
            st.json([msg.dict() for msg in memory.chat_memory.messages])
        elif option == "End":
          st.write("Session ended. Refresh the page to restart")
    except Exception as e:
      st.error(f"Error processing the URL: {str(e)}")
  else:
    st.error("Invalid URL. Please enter a valid website")
