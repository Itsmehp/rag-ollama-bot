# RAG-Bot: PDF Q&A with LangChain, Ollama & Chainlit

This project is a lightweight, fast, and interactive RAG (Retrieval-Augmented Generation) application that allows you to query a **PDF document** using **local LLMs via Ollama**, powered by **LangChain** and **Chainlit**.

---

## Features

- PDF ingestion and chunking
- Embedding with HuggingFace (e5-base-v2)
- Semantic search via Chroma vector store
- LLM-powered answers using Ollama (e.g., `llama3.2`)
- Chat interface using Chainlit (runs in your browser)
- Source citation from original PDF pages

---

## Tech Stack

- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://ollama.com) for running LLMs locally
- [Chainlit](https://github.com/Chainlit/chainlit) for the UI
- [ChromaDB](https://www.trychroma.com/) for vector storage
- HuggingFace embeddings (`intfloat/e5-base-v2`)

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/rag-ollama-bot.git
cd rag-ollama-bot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Ollama and pull your model

Make sure Ollama is installed and running.

```bash
ollama pull llama3.2  # Or your preferred Model
```

Note: If you downloaded a different model, change the model name in app.py here:

```bash
llm = OllamaLLM(
    model="llama3.2",  # Use the model you pulled
    temperature=0.1,
    # num_predict=256,    # Max tokens to generate
)
```

## Usage

### 1. Place your PDF

Place your PDF in the project root directory and update the filename in app.py:

```bash
path = './your-pdf-file.pdf'
```

### 2. Start the Chainlit app

```bash
chainlit run app.py
```

### 3. Ask your questions!

Open the browser link provided (usually http://localhost:8000) and start chatting with your PDF! 📘
