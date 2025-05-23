import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import chainlit as cl

# Load and process the PDF
path = './AIAYN.pdf'
loader = PyPDFLoader(path)
pages = loader.load()

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(pages)

# Create embeddings with explicit model name
embeddings = HuggingFaceEmbeddings(
    # model_name="sentence-transformers/all-mpnet-base-v2"
    model_name="intfloat/e5-base-v2"
)

doc_search = Chroma.from_documents(docs, embeddings)

llm = OllamaLLM(
    model="llama3.2",  # Use the model you pulled
    temperature=0.1,
    # num_predict=256,    # Max tokens to generate
)


@cl.on_chat_start
async def main():
    """Initialize the retrieval chain when chat starts"""
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=doc_search.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    cl.user_session.set("retrieval_chain", retrieval_chain)

    # Send welcome message
    await cl.Message(
        content="🤖 Welcome to RAG-Bot! Upload a PDF and I can answer questions about your PDF document using Ollama. What would you like to know about the content?"
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    """Handle incoming messages and generate responses"""
    try:
        retrieval_chain = cl.user_session.get("retrieval_chain")
        if not retrieval_chain:
            await cl.Message(content="❌ Error: Retrieval chain not initialized.").send()
            return
        # Show loading message
        loading_msg = await cl.Message(content="🔍 Searching through the document...").send()
        # Process the query using ainvoke
        res = await retrieval_chain.ainvoke(
            {"query": message.content},
            config={"callbacks": [cl.AsyncLangchainCallbackHandler()]}
        )
        # Remove loading message and send response
        await loading_msg.remove()
        # Get the answer
        answer = res.get("result", "❌ No answer found.")
        # Optionally show source documents
        sources = res.get("source_documents", [])
        if sources:
            source_text = f"\n\n📚 **Sources used:**\n"
            for i, doc in enumerate(sources[:5], 1):
                source_text += f"{i}. Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:100]}...\n"
            answer += source_text
        await cl.Message(content=answer).send()
    except Exception as e:
        await cl.Message(content=f"❌ An error occurred: {str(e)}").send()
