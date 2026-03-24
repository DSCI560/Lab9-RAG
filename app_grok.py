import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from openai import OpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from htmlTemplates import css, bot_template, user_template

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
CHAT_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct:groq"


#Thin LangChain wrapper around the HF router 
class HFRouterChat(BaseChatModel):
    model_name: str = CHAT_MODEL
    temperature: float = 0.0
    max_tokens: int = 512
    hf_token: str = ""

    @property
    def _llm_type(self) -> str:
        return "hf_router_chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        client = OpenAI(base_url=HF_ROUTER_BASE_URL, api_key=self.hf_token)
        oai_messages = []
        for m in messages:
            if isinstance(m, SystemMessage):
                oai_messages.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage):
                oai_messages.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                oai_messages.append({"role": "assistant", "content": m.content})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=oai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content or ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name}


# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


# Split text into chunks 
def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_text(text)


# Create FAISS vector store 
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


# Build conversation chain
def get_conversation_chain(vectorstore, hf_token):
    llm = HFRouterChat(
        model_name=CHAT_MODEL,
        temperature=0.0,
        max_tokens=512,
        hf_token=hf_token,
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

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
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"])),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: st.session_state.setdefault("msg_history", InMemoryChatMessageHistory()),
        input_messages_key="question",
        history_messages_key="history",
    )
    return chain_with_history


# Handle user input 
def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDFs first!")
        return

    try:
        with st.spinner("Thinking..."):
            answer = st.session_state.conversation.invoke(
                {"question": user_question},
                config={"configurable": {"session_id": "default"}},
            )
    except Exception as exc:
        st.error(f"Question failed: {exc}")
        return

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "bot", "content": str(answer)})

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.write(user_template.replace("{{MSG}}", msg["content"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg["content"]), unsafe_allow_html=True)


#main streamlit app 
def main():
    load_dotenv()

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    if not hf_token:
        st.error(
            "HUGGINGFACEHUB_API_TOKEN is missing. "
            "Get a free token at https://huggingface.co/settings/tokens "
            "and add it to your .env file as HUGGINGFACEHUB_API_TOKEN=hf_..."
        )
        return

    st.set_page_config(page_title="PDF Chat (Llama-3 via HuggingFace)", page_icon="⚡")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "msg_history" not in st.session_state:
        st.session_state.msg_history = InMemoryChatMessageHistory()

    st.header("Chat with PDFs  —  Llama-3 via HuggingFace (Free)")

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.write(user_template.replace("{{MSG}}", msg["content"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg["content"]), unsafe_allow_html=True)

    user_question = st.chat_input("Ask a question about your documents:")
    if user_question and user_question.strip():
        handle_userinput(user_question.strip())

    with st.sidebar:
        st.subheader("Your Documents")
        st.caption(f"Model: {CHAT_MODEL} via HuggingFace (free)")
        pdf_docs = st.file_uploader(
            "Upload PDFs and click Process",
            accept_multiple_files=True,
            type="pdf"
        )

        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
                return

            st.session_state.chat_history = []
            st.session_state.conversation = None
            st.session_state.msg_history = InMemoryChatMessageHistory()

            try:
                with st.spinner("Processing your PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    st.write(f"Extracted {len(raw_text):,} characters")

                    text_chunks = get_text_chunks(raw_text)
                    st.write(f"Created {len(text_chunks)} text chunks")

                    vectorstore = get_vectorstore(text_chunks)
                    st.write("Vector store created")

                    st.session_state.conversation = get_conversation_chain(vectorstore, hf_token)
                    st.success("Ready! Ask me anything about your PDFs.")
            except Exception as exc:
                st.error(f"Processing failed: {exc}")


if __name__ == "__main__":
    main()