import os
import re
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pydantic import Field
from typing import Any, List, Optional, Tuple

from htmlTemplates import css, bot_template, user_template

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Keep defaults CPU-friendly; users can override in .env with OPEN_SOURCE_MODEL_CANDIDATES.
DEFAULT_MODEL_CANDIDATES = "google/flan-t5-base,google/flan-t5-small"


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def load_local_model(model_name: str, use_gpu: bool) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if use_gpu else torch.float32,
    )
    if use_gpu:
        hf_model = hf_model.cuda()
    hf_model.eval()
    return tokenizer, hf_model


#PDF extraction

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text



# chunking

def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    return splitter.split_text(text)



#Vectorstore

def get_vectorstore(text_chunks, progress_bar=None):
    if progress_bar:
        progress_bar.progress(30, text="Loading embedding model...")
    embeddings = load_embedding_model()
    if progress_bar:
        progress_bar.progress(60, text="Creating embeddings...")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    if progress_bar:
        progress_bar.progress(90, text="Vector store ready...")
    return vectorstore



#Repetition post-processor

def remove_repetitions(text: str) -> str:
    if not text:
        return text

    # Remove duplicate consecutive lines first (common in small local model loops).
    line_parts = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if line_parts:
        dedup_lines = [line_parts[0]]
        for ln in line_parts[1:]:
            if ln.lower() != dedup_lines[-1].lower():
                dedup_lines.append(ln)
        text = " ".join(dedup_lines)

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

    cleaned = " ".join(result).strip() or text

    # Trim long repeated phrase loops (e.g., same clause repeated many times).
    loop_match = re.search(r'(.{20,120}?)(?:\s+\1){2,}', cleaned, flags=re.IGNORECASE)
    if loop_match:
        cleaned = loop_match.group(1).strip()

    return cleaned



#Local LLM

class LocalSeq2SeqLLM(LLM):
    model:                Any   = Field(default=None, exclude=True)
    tokenizer:            Any   = Field(default=None, exclude=True)
    use_gpu:              bool  = False
    max_new_tokens:       int   = 96
    num_beams:            int   = 3
    no_repeat_ngram_size: int   = 6
    length_penalty:       float = 1.0
    repetition_penalty:   float = 2.0

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



#Build the conversation chain using LCEL


def get_conversation_chain(vectorstore, progress_bar=None):
    use_gpu = torch.cuda.is_available()
    candidate_models = [
        m.strip() for m in os.getenv("OPEN_SOURCE_MODEL_CANDIDATES", "").split(",") if m.strip()
    ]
    if not candidate_models:
        candidate_models = [
            m.strip() for m in DEFAULT_MODEL_CANDIDATES.split(",") if m.strip()
        ]

    # Try candidate models in order so app remains usable on CPU-only laptops.
    model_load_errors = []
    tokenizer = None
    hf_model = None
    active_model_name = None

    for idx, model_name in enumerate(candidate_models, start=1):
        if progress_bar:
            progress_bar.progress(95, text=f"Loading model {idx}/{len(candidate_models)}: {model_name}")
        try:
            tokenizer, hf_model = load_local_model(model_name=model_name, use_gpu=use_gpu)
            active_model_name = model_name
            break
        except Exception as exc:
            model_load_errors.append(f"{model_name}: {exc}")

    if tokenizer is None or hf_model is None:
        joined_errors = "\n".join(model_load_errors)
        raise RuntimeError(
            "Failed to load any local open-source model. "
            "Set OPEN_SOURCE_MODEL_CANDIDATES in .env to valid model ids.\n"
            f"Errors:\n{joined_errors}"
        )

    llm = LocalSeq2SeqLLM(
        model=hf_model,
        tokenizer=tokenizer,
        use_gpu=use_gpu,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a factual assistant. Use only the provided context to answer. "
            "If the answer is not in the context, say you do not have enough information.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        context_blocks = []
        total_chars = 0
        max_context_chars = 1400

        for d in docs:
            cleaned = " ".join(d.page_content.split())
            remaining = max_context_chars - total_chars
            if remaining <= 0:
                break
            snippet = cleaned[:remaining]
            if snippet:
                context_blocks.append(snippet)
                total_chars += len(snippet)

        return "\n\n".join(context_blocks)

    #LCEL chain- retrieve , format , prompt , llm , parse
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"])),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    #wrap with message history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: st.session_state.setdefault(
            "msg_history",
            InMemoryChatMessageHistory()
        ),
        input_messages_key="question",
        history_messages_key="history",
    )

    return chain_with_history, retriever, active_model_name



#reframe open-ended "summarize" queries

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

    chain, retriever, _active_model_name = st.session_state.conversation

    try:
        with st.spinner("Thinking..."):
            answer = chain.invoke(
                {"question": processed_question},
                config={"configurable": {"session_id": "default"}},
            )
            source_docs = retriever.invoke(processed_question)
    except Exception as exc:
        st.error(f"Question failed: {exc}")
        return

    answer = remove_repetitions(str(answer))
    if not answer or len(answer) < 5:
        answer = "No answer could be generated. Try rephrasing your question."

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "bot",  "content": answer})

    #render full conversation
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



# 9. Main

def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Chat (Open Source)", page_icon="🦙")
    st.write(css, unsafe_allow_html=True)

    # Initialise session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "msg_history" not in st.session_state:
        st.session_state.msg_history = InMemoryChatMessageHistory()
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0

    st.header("Chat with PDFs - Open Source")
    if st.session_state.conversation is not None:
        _chain, _retriever, active_model_name = st.session_state.conversation
        st.caption(f"Active local model: {active_model_name}")

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
    user_question = st.chat_input("Ask a question about your documents:")

    if user_question and user_question.strip():
        handle_userinput(user_question.strip())

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs and click Process",
            accept_multiple_files=True,
            type="pdf",
        )

        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
                return

            st.session_state.chat_history = []
            st.session_state.conversation = None
            st.session_state.msg_history = InMemoryChatMessageHistory()
            st.session_state.last_question = ""

            try:
                progress = st.progress(0, text="Starting...")

                progress.progress(10, text="Extracting text from PDFs...")
                raw_text = get_pdf_text(pdf_docs)
                st.info(f"Extracted {len(raw_text):,} characters")

                if not raw_text.strip():
                    st.error("No extractable text found in the uploaded PDFs.")
                    return

                progress.progress(20, text="Splitting into chunks...")
                chunks = get_text_chunks(raw_text)
                st.info(f"Created {len(chunks)} chunks")

                if not chunks:
                    st.error("Chunking returned zero chunks. Check PDF text extraction.")
                    return

                vectorstore = get_vectorstore(chunks, progress_bar=progress)
                st.info("Vector store built")

                st.session_state.conversation = get_conversation_chain(
                    vectorstore, progress_bar=progress
                )

                progress.progress(100, text="Done!")
                _chain, _retriever, active_model_name = st.session_state.conversation
                st.success(f"Ready. Ask a question (model: {active_model_name}).")
            except Exception as exc:
                st.error(f"Processing failed: {exc}")


if __name__ == "__main__":
    main()