# Personal AI Knowledge Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content using natural language.

## Features

- Upload PDF documents.
- Extract and split document text into smaller chunks.
- Generate semantic embeddings using Sentence Transformers.
- Store and retrieve relevant text using ChromaDB.
- Generate answers using Ollama with the Llama 3.2 model.
- Display source information to help users verify answers.

## Technologies Used

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- Ollama
- Llama 3.2
- pypdf
- Requests

## How It Works

1. Upload a PDF document.
2. Extract text and divide it into chunks.
3. Convert the chunks into embeddings.
4. Store the embeddings in ChromaDB.
5. Retrieve relevant chunks for a user's question.
6. Pass the retrieved context to the language model.
7. Display the generated answer and available source information.

## Run Locally

1. Install Python and Git.
2. Install Ollama and download the model:

   `ollama pull llama3.2`

3. Install the required Python packages:

   `pip install -r requirements.txt`

4. Start the application:

   `streamlit run app.py`

## Project Goal

To build a personal knowledge assistant that uses semantic search and Retrieval-Augmented Generation to answer questions from PDF documents.
