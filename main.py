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
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import validators

# Load OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["key"]

# Streamlit user interface
st.title("Conversational Website Chatbot")
st.write("Enter a valid website to start processing its contents")

# Initialize memory in session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

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

            # Setup LLM
            llm = ChatOpenAI(temperature=0.2)

            # Combine database, memory, and LLM
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectordb.as_retriever(),
                memory=st.session_state.memory,
                get_chat_history=lambda h :h,
            )

            # Provide options after processing
            st.write("Content processed successfully")
            option = st.radio(
                "What do you like to do next?",
                options=["Start Chatting", "End"],
                index=0
            )

            # If start chatting
            if option == "Start Chatting":
                st.write("Chatbot is ready! Start asking questions.")
                user_input = st.text_area("Your question (type 'exit' if you're done asking): ", key="user_input")

                if user_input:
                    if user_input.lower() == "exit":
                        st.write("Exiting. Refresh the page to restart.")
                    else:
                        # Use session memory for chat history
                        response = chain.invoke({
                            "messages": st.session_state.memory.chat_memory.messages,
                            "question": user_input
                        })["answer"]

                        st.write(f"Chatbot: {response}")

                        # Add new user and AI messages to memory
                        st.session_state.memory.chat_memory.add_user_message(user_input)
                        st.session_state.memory.chat_memory.add_ai_message(response)

                        # Debugging: Display full memory content
                        st.write("### Debug: Memory Content")
                        st.json([msg.dict() for msg in st.session_state.memory.chat_memory.messages])
            elif option == "End":
                st.write("Session ended. Refresh the page to restart.")
        except Exception as e:
            st.error(f"Error processing the URL: {str(e)}")
    else:
        st.error("Invalid URL. Please enter a valid website.")
