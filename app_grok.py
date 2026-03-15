"""
app_groq.py  —  PDF Chatbot using Groq (FREE, fast) + OpenAI Embeddings fallback
===================================================================================
Groq gives you FREE access to Llama-3, Mixtral etc. with NO credit card needed.

Setup:
1. Go to https://console.groq.com  →  sign up free  →  API Keys  →  Create Key
2. Add to your .env file:
       GROQ_API_KEY=your_groq_key_here
3. pip install streamlit pypdf2 langchain langchain-community langchain-groq \
              python-dotenv faiss-cpu sentence_transformers tiktoken
4. streamlit run app_groq.py
"""

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings   # free embeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq                                  # free LLM via Groq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


# ── 1. Extract text from PDFs ─────────────────────────────────────────────────
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


# ── 2. Split text into chunks ─────────────────────────────────────────────────
def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_text(text)


# ── 3. Create FAISS vector store (free MiniLM embeddings) ────────────────────
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# ── 4. Build conversation chain using Groq ───────────────────────────────────
def get_conversation_chain(vectorstore):
    """
    Groq hosts Llama-3 / Mixtral for FREE with very fast inference.
    Model options (all free on Groq):
      - "llama3-8b-8192"       ← fast, good quality
      - "llama3-70b-8192"      ← slower, best quality
      - "mixtral-8x7b-32768"   ← great for long documents
    """
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        ),
        memory=memory,
    )
    return chain


# ── 5. Handle user question and render chat ───────────────────────────────────
def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("⚠️ Please upload and process PDFs first!")
        return

    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(
            template.replace("{{MSG}}", message.content),
            unsafe_allow_html=True
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Chat (Groq)", page_icon="⚡")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("⚡ Chat with PDFs — Powered by Groq (Free & Fast)")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("📄 Your Documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs and click 'Process'",
            accept_multiple_files=True,
            type="pdf"
        )

        if st.button("Process"):
            if not pdf_docs:
                st.error("upload PDF files")
                return

            with st.spinner("Processing your PDFs..."):
                # 1. Extract text
                raw_text = get_pdf_text(pdf_docs)
                st.write(f"Extracted {len(raw_text)} characters")

                # 2. Split into chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(f"Created {len(text_chunks)} text chunks")

                # 3. Create vector store
                vectorstore = get_vectorstore(text_chunks)
                st.write("Vector store created")

                # 4. Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Ask question")


if __name__ == "__main__":
    main()