# app_opensource.py
# Compatible with langchain>=1.0, langchain-community, langchain-core, langchain-text-splitters
# All legacy APIs (langchain.memory, langchain.chains, langchain.llms.base) are REMOVED in 1.x.
# This script uses the modern LCEL (LangChain Expression Language) equivalents.

import os
import re
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# ── Text splitter (own package since langchain 0.1) ───────────────────────────
from langchain_text_splitters import CharacterTextSplitter

# ── Community: embeddings + vector store ─────────────────────────────────────
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── langchain_core: everything else ──────────────────────────────────────────
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pydantic import Field
from typing import Any, List, Optional

from htmlTemplates import css, bot_template, user_template


# ─────────────────────────────────────────────────────────────────────────────
# 1. PDF extraction
# ─────────────────────────────────────────────────────────────────────────────
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 2. Chunking
# ─────────────────────────────────────────────────────────────────────────────
def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    return splitter.split_text(text)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Vectorstore
# ─────────────────────────────────────────────────────────────────────────────
def get_vectorstore(text_chunks, progress_bar=None):
    if progress_bar:
        progress_bar.progress(30, text="Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    if progress_bar:
        progress_bar.progress(60, text="Creating embeddings...")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    if progress_bar:
        progress_bar.progress(90, text="Vector store ready...")
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# 4. Repetition post-processor
# ─────────────────────────────────────────────────────────────────────────────
def remove_repetitions(text: str) -> str:
    if not text:
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    seen = []
    result = []
    for s in sentences:
        norm = s.strip().lower()
        if norm and norm not in seen:
            seen.append(norm)
            result.append(s.strip())
        elif norm in seen:
            break
    return " ".join(result).strip() or text


# ─────────────────────────────────────────────────────────────────────────────
# 5. Local LLM — inherits from langchain_core.language_models.llms.LLM
#    (langchain.llms.base.LLM is gone in langchain 1.x)
# ─────────────────────────────────────────────────────────────────────────────
class LocalSeq2SeqLLM(LLM):
    model:                Any   = Field(default=None, exclude=True)
    tokenizer:            Any   = Field(default=None, exclude=True)
    use_gpu:              bool  = False
    max_new_tokens:       int   = 150
    num_beams:            int   = 4
    no_repeat_ngram_size: int   = 5
    length_penalty:       float = 0.8
    repetition_penalty:   float = 3.5

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "local_seq2seq"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        if self.use_gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                num_beam_groups=1,        # explicitly override any cached generation_config
                do_sample=False,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                repetition_penalty=self.repetition_penalty,
                length_penalty=self.length_penalty,
                early_stopping=True,
            )

        raw = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        cleaned = remove_repetitions(raw)
        if not cleaned or len(cleaned) < 5:
            return "I could not generate a confident answer from the provided context."
        return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# 6. Build the conversation chain using LCEL
#    Replaces: ConversationalRetrievalChain, ConversationBufferMemory, LLMChain
#    Uses:     RunnableWithMessageHistory + InMemoryChatMessageHistory
# ─────────────────────────────────────────────────────────────────────────────
def get_conversation_chain(vectorstore, progress_bar=None):
    if progress_bar:
        progress_bar.progress(95, text="Loading language model (flan-t5-xl)...")

    use_gpu = torch.cuda.is_available()
    # flan-t5-xl (3B) — better instruction following than flan-t5-large (770M)
    # Use "google/flan-t5-large" if you have less than 8 GB RAM free
    model_name = "google/flan-t5-xl"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if use_gpu else torch.float32,
    )
    if use_gpu:
        hf_model = hf_model.cuda()
    hf_model.eval()

    llm = LocalSeq2SeqLLM(
        model=hf_model,
        tokenizer=tokenizer,
        use_gpu=use_gpu,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    # Prompt that includes retrieved context + message history
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a factual assistant. Use ONLY the context below to answer.\n"
         "Rules:\n"
         "- Answer in 1 to 3 sentences maximum.\n"
         "- Do NOT repeat any sentence.\n"
         "- Do NOT add information outside the context.\n"
         "- If the answer is not in the context, say: "
         "'The documents do not contain enough information to answer this question.'\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # LCEL chain: retrieve → format → prompt → llm → parse
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"])),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Wrap with message history (replaces ConversationBufferMemory)
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: st.session_state.setdefault(
            "msg_history",
            InMemoryChatMessageHistory()
        ),
        input_messages_key="question",
        history_messages_key="history",
    )

    return chain_with_history, retriever


# ─────────────────────────────────────────────────────────────────────────────
# 7. Reframe open-ended "summarize" queries
# ─────────────────────────────────────────────────────────────────────────────
SUMMARIZE_RE = re.compile(
    r'\b(summarize|summary|summarise|give me an overview|'
    r'what is this (document|pdf|about)|'
    r'what does this (document|pdf) (say|cover|contain))\b',
    re.IGNORECASE,
)

def preprocess_question(question: str) -> str:
    if SUMMARIZE_RE.search(question):
        return (
            "What is the main topic of the document and "
            "what are the key steps or concepts it describes?"
        )
    return question


# ─────────────────────────────────────────────────────────────────────────────
# 8. Handle user input
# ─────────────────────────────────────────────────────────────────────────────
def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDFs first!")
        return

    processed_question = preprocess_question(user_question)
    if processed_question != user_question:
        st.info(
            f"Rephrased to: *{processed_question}*  \n"
            "(flan-t5 works best with specific factual questions)"
        )

    chain, retriever = st.session_state.conversation

    with st.spinner("Thinking..."):
        answer = chain.invoke(
            {"question": processed_question},
            config={"configurable": {"session_id": "default"}},
        )
        source_docs = retriever.invoke(processed_question)

    answer = remove_repetitions(str(answer))
    if not answer or len(answer) < 5:
        answer = "No answer could be generated. Try rephrasing your question."

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "bot",  "content": answer})

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.write(
                user_template.replace("{{MSG}}", msg["content"]),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", msg["content"]),
                unsafe_allow_html=True,
            )

    if source_docs:
        with st.expander("Source chunks used to answer"):
            for j, doc in enumerate(source_docs):
                st.markdown(f"**Chunk {j+1}:**")
                st.text(doc.page_content[:400] + "...")
                st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Chat (Open Source)", page_icon="🦙")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "msg_history" not in st.session_state:
        st.session_state.msg_history = InMemoryChatMessageHistory()

    st.header("Chat with PDFs - Open Source (flan-t5-xl)")
    st.caption(
        "Tip: Ask specific factual questions like "
        "'What is the first step to create a workspace?' "
        "rather than 'Summarize'."
    )

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs and click Process",
            accept_multiple_files=True,
            type="pdf",
        )

        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF!")
                return

            st.session_state.chat_history = []
            st.session_state.conversation = None
            st.session_state.msg_history = InMemoryChatMessageHistory()

            progress = st.progress(0, text="Starting...")

            progress.progress(10, text="Extracting text from PDFs...")
            raw_text = get_pdf_text(pdf_docs)
            st.info(f"Extracted {len(raw_text):,} characters")

            progress.progress(20, text="Splitting into chunks...")
            chunks = get_text_chunks(raw_text)
            st.info(f"Created {len(chunks)} chunks")

            vectorstore = get_vectorstore(chunks, progress_bar=progress)
            st.info("Vector store built!")

            st.session_state.conversation = get_conversation_chain(
                vectorstore, progress_bar=progress
            )

            progress.progress(100, text="Done!")
            st.success("Ready! Ask me anything about your PDFs.")


if __name__ == "__main__":
    main()