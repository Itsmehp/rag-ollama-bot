import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import chainlit as cl

# Create embeddings with explicit model name
embeddings = HuggingFaceEmbeddings(
    # model_name="sentence-transformers/all-mpnet-base-v2"
    model_name="intfloat/e5-base-v2"
)

llm = OllamaLLM(
    model="llama3.2",  # Use the model you pulled
    temperature=0.1,
    # num_predict=256,    # Max tokens to generate
)


@cl.on_chat_start
async def main():
    await cl.Message(
        content="📄 Please upload a PDF file to get started."
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    try:
        # Handle file uploads
        if message.elements:
            file = message.elements[0]  # Take first uploaded file
            loader = PyPDFLoader(file.path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100)
            docs = splitter.split_documents(pages)

            # Embeddings + Vector DB
            embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/e5-base-v2")
            doc_search = Chroma.from_documents(docs, embeddings)

            # Retrieval chain
            llm = OllamaLLM(model="llama3.2", temperature=0.1)
            retrieval_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=doc_search.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )

            cl.user_session.set("retrieval_chain", retrieval_chain)

            await cl.Message(content="✅ File processed! Ask me a question.").send()
            return

        # Continue with question-answering
        retrieval_chain = cl.user_session.get("retrieval_chain")
        if not retrieval_chain:
            await cl.Message(content="📁 Please upload a PDF before asking questions.").send()
            return

        loading_msg = await cl.Message(content="🔍 Searching through the document...").send()

        res = await retrieval_chain.ainvoke(
            {"query": message.content},
            config={"callbacks": [cl.AsyncLangchainCallbackHandler()]}
        )

        await loading_msg.remove()

        answer = res.get("result", "❌ No answer found.")
        sources = res.get("source_documents", [])
        if sources:
            source_text = f"\n\n📚 **Sources used:**\n"
            for i, doc in enumerate(sources[:5], 1):
                source_text += f"{i}. Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:100]}...\n"
            answer += source_text

        await cl.Message(content=answer).send()

    except Exception as e:
        await cl.Message(content=f"❌ An error occurred: {str(e)}").send()
