import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time

# Function to set up embeddings with the provided NVIDIA API key
def vector_embedding(api_key):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings(api_key=api_key)
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        
        if not st.session_state.docs:
            st.error("No documents loaded. Please check the directory or files.")
            return

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])  # Splitting
        
        if not st.session_state.final_documents:
            st.error("No document chunks created. Please check the text splitter settings.")
            return

        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings
        st.success("Vector store created successfully!")

st.title("Nvidia NIM Demo")

# Input field for the NVIDIA API key
api_key = st.text_input("Enter your NVIDIA API Key", type="password")

# Input field for the user's question
prompt1 = st.text_input("Enter Your Question From Documents")

# Button to create the vector embeddings
if st.button("Create Vector Store"):
    if api_key:
        vector_embedding(api_key)
    else:
        st.warning("Please enter your NVIDIA API Key.")

if prompt1 and api_key and "vectors" in st.session_state:
    llm = ChatNVIDIA(api_key=api_key, model="meta/llama3-70b-instruct")
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write("Response time:", time.process_time() - start)
    st.write(response['answer'])

    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
