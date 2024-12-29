# Medium Website Chatbot

## Overview

The Medium Website Chatbot is a Streamlit-based application that allows users to engage in conversational queries about the content of any provided medium blogs found [here](https://medium.com/). Powered by OpenAI's language models, it extracts, processes, and retrieves information from the website to facilitate an intelligent chatbot experience.

## Features

- **Conversational Interface**: Chat in natural language about the content of a website.
- **Dynamic Website Processing**: Load and analyze content from any valid URL.
- **Powered by OpenAI**: Uses OpenAI's embeddings and language models for document retrieval and conversational understanding.
- **Session Memory**: Maintains chat history for context-aware responses.

## How It Works

1. Enter a valid medium website URL.
2. The chatbot extracts and processes the content from the website.
3. It builds a conversational retrieval model using OpenAI embeddings and FAISS vector databases.
4. Users can then ask questions or have a conversation about the website's content.

## Requirements

### Prerequisites

- Python 3.8+
- OpenAI API key
- Streamlit
- Required Python libraries: langchain, streamlit, validators, faiss-cpu, and openai
