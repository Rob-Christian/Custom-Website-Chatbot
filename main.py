# Import necessary packages
import os
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
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
            loaders = UnstructuredURLLoader(urls=[url])
            data = loaders.load()

            # Extract chunks from the website
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(data)

            # Generate embeddings and create FAISS database
            embeddings = OpenAIEmbeddings()
            vectordb = FAISS.from_documents(documents=chunks, embedding=embeddings)

            # Setup LLM and memory
            llm = ChatOpenAI(temperature=0)
            memory = ConversationBufferMemory(
                memory_key = "chat_history",
                return_messages = True,
                output_key = "answer"
            )

            # Setup prompt
            prompt_template = """
            You are a helpful website chatbot assistant. Use the retrieved information and your past conversation to answer the conversational questions.
            {context}

            {chat_history}
            
            Question: {question}
            Helpful Answer:
            """
            prompt = PrompTemplate(template = prompt_template, input_variables = ["context", "chat_history", "question"])

            # Combine LLM, memory, and prompt
            chain = ConversationalRetrievalChain.from_llm(
                llm = llm,
                retriever = vectordb.as_retriever(search_kwargs = {"k":1}),
                memory = memory,
                get_chat_history = lambda h: h,
                combine_docs_chain_kwargs = {'prompt': prompt}
            )

            # Provide options after processing
            st.write("Content processed successfully")
            st.write("Website Chatbot Assistant is ready! Start asking questions.")

            while True:
                user_input = st.text_area("Your question (type 'exit' if you're done asking): ", key="user_input")

                if user_input:
                    if user_input.lower() == "exit":
                        st.write("Exiting. Refresh the page to restart")
                    else:
                        response = chain({"question": user_input})
                        st.write(f"Chatbot: {response["answer"]}")
                        
        except Exception as e:
            st.error(f"Error processing the URL: {str(e)}")
    else:
        st.error("Invalid URL. Please enter a valid website.")
