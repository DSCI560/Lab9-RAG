# Lab 9 – Custom Q&A Chatbot with Retrieval Augmented Generation (RAG)

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** that allows users to ask questions about uploaded PDF documents.

The system extracts text from PDFs, splits it into manageable chunks, generates embeddings, stores them in a **FAISS vector database**, and retrieves the most relevant chunks when answering user questions. The retrieved context is then passed to a **Large Language Model (LLM)** to generate accurate responses.

The application is built with **Streamlit**, enabling users to upload PDFs, process them, and interact with the chatbot through a web interface.

This project was developed for **DSCI 560 – Data Science Practicum Lab 9**. 

---

# System Architecture

The chatbot follows the **RAG pipeline**:

1. **PDF Upload**
2. **Text Extraction**
3. **Text Chunking**
4. **Embedding Generation**
5. **Vector Storage (FAISS)**
6. **Similarity Search**
7. **LLM Response Generation**

According to the lab instructions, the system should:

* Convert PDFs into text
* Split the text into chunks
* Create embeddings for each chunk
* Store embeddings in a vector database
* Retrieve relevant chunks for user queries
* Generate answers using an LLM 

---

# Project Structure

```
Lab9-RAG
│
├── app_chatgpt.py        # RAG chatbot using OpenAI (GPT-3.5)
├── app_grok.py           # Free chatbot using Groq Llama-3
├── app_opensource.py     # Fully local open-source chatbot (Flan-T5)
│
├── vector_db_monitor.py  # Tool to inspect and debug the FAISS vector database
│
├── htmlTemplates.py      # HTML + CSS templates for chat interface
├── requirements.txt      # Python dependencies
│
└── lab9.pdf              # Lab instructions
```

---

# Implementations

This project contains **three different chatbot implementations**.

## 1. OpenAI Version

File:

```
app_chatgpt.py
```

Features:

* Uses **OpenAI embeddings**
* Uses **GPT-3.5 Turbo**
* FAISS vector database
* Conversational memory

Pipeline:

```
PDF -> text -> chunks -> OpenAI embeddings -> FAISS -> GPT-3.5 answer
```

The chatbot retrieves the **top 4 most relevant chunks** from the vector store before generating the answer. 

---

## 2. Groq Version (Free)

File:

```
app_grok.py
```

Features:

* **Groq API (free)**
* Llama-3 models
* HuggingFace embeddings
* FAISS vector search

Supported Groq models include:

* `llama3-8b-8192`
* `llama3-70b-8192`
* `mixtral-8x7b-32768`

Groq provides **very fast inference without OpenAI costs**. 

---

## 3. Fully Open-Source Version

File:

```
app_opensource.py
```

Features:

* Local **Flan-T5-XL (3B)** model
* HuggingFace embeddings
* FAISS vector store
* Local inference using **Transformers + PyTorch**

This version does **not require external APIs** and keeps all data local.

Additional improvements include:

* repetition filtering
* improved prompts
* message history tracking
* context-restricted answers

The model only answers using retrieved document context.

---

# Vector Database Monitor

File:

```
vector_db_monitor.py
```

This tool allows users to inspect the vector database.

Features:

* Browse all stored text chunks
* View chunk size statistics
* Visualize chunk distribution
* Test similarity search queries

The FAISS index is saved locally and can be loaded for inspection. 

---

# Web Interface

The chatbot interface is built with **Streamlit** and custom HTML templates.

Features:

* Upload one or multiple PDFs
* Process documents
* Ask questions in chat format
* View conversation history

The interface styling is implemented using **custom CSS and HTML templates**. 

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/DSCI560/Lab9-RAG.git
cd Lab9-RAG
```

---

## 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:

* Streamlit
* LangChain
* FAISS
* HuggingFace Transformers
* Sentence Transformers
* PyTorch 

---

# Environment Variables

Create a `.env` file in the project root.

### OpenAI version

```
OPENAI_API_KEY=your_openai_key
```

### Groq version

```
GROQ_API_KEY=your_groq_key
```

---

# Running the Chatbot

## OpenAI Version

```
streamlit run app_chatgpt.py
```

---

## Groq Version (Free)

```
streamlit run app_grok.py
```

---

## Open-Source Version (Local)

```
streamlit run app_opensource.py
```

---

# Using the Chatbot

1. Upload one or more PDF files

2. Click **Process**

3. The system will:

   * Extract text
   * Split text into chunks
   * Generate embeddings
   * Build a FAISS vector database

4. Ask questions about the documents.

The chatbot retrieves relevant chunks and generates answers based on those sections.

---

# Example Questions

* What is the main topic of the document?
* How do I install the software described in the PDF?
* What are the key steps mentioned in the guide?
* Summarize the main ideas of the document.

---

# Technologies Used

* Python
* Streamlit
* LangChain
* FAISS
* HuggingFace Transformers
* Sentence Transformers
* PyTorch
* OpenAI API
* Groq API

---


