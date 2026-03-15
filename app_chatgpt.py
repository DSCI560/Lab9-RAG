import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
# from langchain_community.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferMemory
from htmlTemplates import css, bot_template, user_template
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    #Loop through each PDF and extract all text into one big string
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# split text into chunks 
def get_text_chunks(text):
    
    #Split the raw text into overlapping chunks of 500 characters.
    #Overlap of 100 ensures context isn't lost at chunk boundaries.
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# creating vector store from chunks 
def get_vectorstore(text_chunks):
    
    # Convert each chunk into an embedding vector using OpenAI,
    # then store all vectors in a FAISS vector database for fast search.
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


#  building conversation chain 
def get_conversation_chain(vectorstore):
    
    #ChatOpenAI as the LLM
    #The vector store as the retriever (finds top 4 relevant chunks)
    #ConversationBufferMemory to remember chat history
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}   # retrieve top 4 most relevant chunks
        ),
        memory=memory,
    )
    return conversation_chain


# handle user input and display chat 
def handle_userinput(user_question):
    
    # Send the user's question through the conversation chain,
    
    if st.session_state.conversation is None:
        st.warning("Please upload and process your PDFs first!")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    #even indexed messages = user, odd-indexed = bot
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True
            )

def main():
    load_dotenv()

    st.set_page_config(
        page_title="Chat with PDFs",
        page_icon=":robot_face:"
    )
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with PDFs")

    # Chat input
    user_question = st.text_input("question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # Sidebar for PDF upload
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click 'Process'",
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


if __name__ == '__main__':
    main()