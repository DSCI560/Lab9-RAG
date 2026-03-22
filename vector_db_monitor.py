import streamlit as st
import os
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter  
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

vecPath = "vectorstore_index"


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def save_vectorstore(vectorstore):
    vectorstore.save_local(vecPath)
    st.success(f"Vector store saved to `{vecPath}/`")


def load_vectorstore(embeddings):
    if os.path.exists(vecPath):
        return FAISS.load_local(
            vecPath,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None


def get_all_chunks(vectorstore):
    docstore = vectorstore.docstore
    chunks = []
    for doc_id in vectorstore.index_to_docstore_id.values():
        doc = docstore.search(doc_id)
        if doc:
            chunks.append(doc.page_content)
    return chunks


def main():
    st.set_page_config(
        page_title="Vector DB Monitor",
        page_icon="🔬",
        layout="wide"
    )

    st.title("Vector Store Monitor")
    st.caption("Inspect your FAISS vector database in real-time")

    embeddings = load_embeddings()

    with st.sidebar:
        st.header("Build Vector Store")
        st.caption("Upload PDFs here to build & save a vector store for monitoring")

        pdf_docs = st.file_uploader(
            "Upload PDFs",
            accept_multiple_files=True,
            type="pdf"
        )

        if st.button("Build & Save Vector Store"):
            if not pdf_docs:
                st.error("Upload at least one PDF!")
            else:
                with st.spinner("Building..."):
                    text = ""
                    for pdf in pdf_docs:
                        reader = PdfReader(pdf)
                        for page in reader.pages:
                            extracted = page.extract_text()
                            if extracted:
                                text += extracted

                    splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=500,
                        chunk_overlap=100,
                        length_function=len
                    )
                    chunks = splitter.split_text(text)

                    vs = FAISS.from_texts(texts=chunks, embedding=embeddings)
                    save_vectorstore(vs)
                    st.session_state.vectorstore = vs
                    st.rerun()

        st.divider()

        if st.button("Load Saved Vector Store"):
            vs = load_vectorstore(embeddings)
            if vs:
                st.session_state.vectorstore = vs
                st.success("Loaded!")
                st.rerun()
            else:
                st.error(f"No saved store found at `{vecPath}/`")

    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        vs = load_vectorstore(embeddings)
        if vs:
            st.session_state.vectorstore = vs
        else:
            st.info("Upload PDFs in the sidebar")
            return

    vs = st.session_state.vectorstore
    chunks = get_all_chunks(vs)

    col1, col2, col3, col4 = st.columns(4)
    chunk_lengths = [len(c) for c in chunks]

    col1.metric("Total Chunks", len(chunks))
    col2.metric("Avg Chunk Size", f"{int(np.mean(chunk_lengths))} chars")
    col3.metric("Max Chunk Size", f"{max(chunk_lengths)} chars")
    col4.metric("Min Chunk Size", f"{min(chunk_lengths)} chars")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["browse chunks", "size distribution", "similarity search"])

    with tab1:
        st.subheader("All Stored Text Chunks")
        search_filter = st.text_input("Filter chunks by keyword:", "")

        filtered = [
            (i, c) for i, c in enumerate(chunks)
            if search_filter.lower() in c.lower()
        ] if search_filter else list(enumerate(chunks))

        st.caption(f"Showing {len(filtered)} of {len(chunks)} chunks")

        for i, chunk in filtered:
            with st.expander(f"Chunk {i+1}  ({len(chunk)} chars)"):
                st.text(chunk)

    with tab2:
        st.subheader("Chunk Size Distribution")

        import pandas as pd
        df = pd.DataFrame({
            "Chunk Index": range(len(chunks)),
            "Size (chars)": chunk_lengths
        })

        st.bar_chart(df.set_index("Chunk Index")["Size (chars)"])

        st.subheader("Size Breakdown")
        bins = {"< 200": 0, "200-400": 0, "400-500": 0, "500-700": 0, "> 700": 0}
        for l in chunk_lengths:
            if l < 200:
                bins["< 200"] += 1
            elif l < 400:
                bins["200-400"] += 1
            elif l < 500:
                bins["400-500"] += 1
            elif l < 700:
                bins["500-700"] += 1
            else:
                bins["> 700"] += 1

        bin_df = pd.DataFrame(list(bins.items()), columns=["Range", "Count"])
        st.dataframe(bin_df, use_container_width=True)

    with tab3:
        st.subheader("Test Similarity Search")
        st.caption("Type any question to see which chunks the model would retrieve")

        query = st.text_input("Enter a test query:", "How do I install ADS on Linux?")
        k = st.slider("Number of chunks to retrieve (k):", 1, 10, 4)

        if st.button("Search") and query:
            with st.spinner("Searching..."):
                results = vs.similarity_search_with_score(query, k=k)

            st.success(f"Top {k} most relevant chunks for: \"{query}\"")
            for rank, (doc, score) in enumerate(results):
                similarity_pct = max(0, 100 - score * 10)
                st.markdown(f"**#{rank+1} - Distance Score: `{score:.4f}`**")
                progress_value = float(min(1.0, similarity_pct / 100))
                st.progress(progress_value)
                with st.expander(f"View chunk ({len(doc.page_content)} chars)"):
                    st.text(doc.page_content)


if __name__ == "__main__":
    main()