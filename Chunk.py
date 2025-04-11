import os import streamlit as st import base64 from langchain.embeddings import HuggingFaceEmbeddings from langchain.vectorstores import FAISS from langchain.text_splitter import RecursiveCharacterTextSplitter from langchain.llms import ChatGroq from langchain.chains import RetrievalQA from langchain.document_loaders import TextLoader from langchain.schema import Document

Configurations

DATA_DIR = "data" INDEX_DIR = "faiss_index" UPLOAD_PATH = "uploaded_code.txt" CHUNK_SIZE = 1500 CHUNK_OVERLAP = 300 SEPARATORS = ["\n", "<div", "<section", "<style", "<script", "}", "{"] MODEL_NAME = "llama3-8b-8192"

Load all .css files from the data folder

def load_css_documents(folder_path=DATA_DIR): css_docs = [] for file in os.listdir(folder_path): if file.endswith(".css"): with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f: content = f.read() css_docs.append(Document(page_content=content, metadata={"source": file})) return css_docs

Load the uploaded Penpot file

def load_uploaded_file(uploaded_file): with open(UPLOAD_PATH, "wb") as f: f.write(uploaded_file.getvalue()) loader = TextLoader(UPLOAD_PATH) return loader.load()

Chunk documents

def split_documents(documents): splitter = RecursiveCharacterTextSplitter( chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=SEPARATORS ) return splitter.split_documents(documents)

Create and save vectorstore

def embed_documents(docs, persist_path=INDEX_DIR): embeddings = HuggingFaceEmbeddings() vectorstore = FAISS.from_documents(docs, embeddings) vectorstore.save_local(persist_path) return vectorstore

Load existing vectorstore

def load_vectorstore(persist_path=INDEX_DIR): embeddings = HuggingFaceEmbeddings() return FAISS.load_local(persist_path, embeddings)

Get LLM (Groq)

def get_llm(): return ChatGroq(model=MODEL_NAME)

Streamlit App

st.set_page_config(page_title="Penpot to Bootstrap - AI Assistant", layout="wide") st.title("Penpot Design to Bootstrap Converter with RAG")

uploaded_file = st.file_uploader("Upload Penpot HTML or CSS", type=["html", "css"]) query = st.text_input("Ask a question or request a Bootstrap conversion:")

if uploaded_file: with st.spinner("Processing uploaded file and CSS framework..."): uploaded_docs = load_uploaded_file(uploaded_file) css_docs = load_css_documents() all_docs = uploaded_docs + css_docs chunks = split_documents(all_docs)

vectorstore = embed_documents(chunks)
    retriever = vectorstore.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=retriever,
        return_source_documents=True
    )

if query:
    with st.spinner("Generating response..."):
        response = qa.run(f"Generate a complete HTML document with embedded CSS only from the data, no external links or class names. Question: {query}")
        st.subheader("Response:")
        st.markdown(response)

        # Live preview + download button
        preview_file = "preview.html"
        with open(preview_file, "w", encoding="utf-8") as f:
            f.write(response)

        with open(preview_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:text/html;base64,{b64}" target="_blank">Live Preview</a>'
            st.markdown(href, unsafe_allow_html=True)

else: st.info("Please upload a Penpot HTML or CSS file to begin.")

