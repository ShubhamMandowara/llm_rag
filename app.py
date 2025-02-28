import streamlit as st
import fitz  # PyMuPDF for text extraction
import pdfplumber  # For table extraction
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import uuid  # For unique file naming

hf_api_key = st.sidebar.text_input("🔑 Enter Hugging Face API Key", type="password")

# Streamlit UI
st.title("📖 PDF Q&A Chatbot")
st.write("Upload a PDF and ask questions based on its content!")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if not hf_api_key:
    st.warning("Please enter your Hugging Face API key to proceed.")
    st.stop()
else:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key

# Initialize session state for unique user isolation
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # Unique ID for each user session

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = None

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Function to extract tables from PDF
def extract_tables_from_pdf(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_table = page.extract_table()
            if extracted_table:
                tables.append("\n".join(["\t".join(row) for row in extracted_table]))  # Convert table to text
    return tables

# Function to store text embeddings in FAISS
def create_faiss_index(text_data):
    text_chunks = text_data.split("\n\n")  # Split text into chunks
    embeddings = np.array([embedding_model.encode(chunk) for chunk in text_chunks])

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    return index, text_chunks

# Function to retrieve relevant text
def retrieve_relevant_text(query, index, text_chunks, top_k=3):
    query_embedding = np.array([embedding_model.encode(query)])
    distances, indices = index.search(query_embedding, top_k)
    
    return [text_chunks[i] for i in indices[0]]

# Load LLM
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.3", model_kwargs={"temperature": 0.5})

# Define prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Answer based on the context below:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
)

qa_chain = LLMChain(llm=llm, prompt=prompt)

# Process uploaded PDF
if uploaded_file and hf_api_key:
    st.write("✅ PDF uploaded successfully!")

    # Create a unique file path for the user's session
    pdf_path = f"uploaded_{st.session_state.session_id}.pdf"
    
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract content
    extracted_text = extract_text_from_pdf(pdf_path)
    extracted_tables = extract_tables_from_pdf(pdf_path)

    # Combine text and tables
    all_text_data = extracted_text + "\n".join(extracted_tables)

    # Create FAISS index (store it per user session)
    index, text_chunks = create_faiss_index(all_text_data)
    
    st.session_state.faiss_index = index
    st.session_state.text_chunks = text_chunks

    st.success("✅ PDF processed! You can now ask questions.")

# User input for Q&A
user_question = st.text_input("Ask a question about the PDF:")
if user_question and st.session_state.faiss_index:
    relevant_chunks = retrieve_relevant_text(user_question, st.session_state.faiss_index, st.session_state.text_chunks)
    context = "\n".join(relevant_chunks)

    # Generate answer
    answer = qa_chain.run({"context": context, "question": user_question})

    st.write("### Answer:")
    st.write(answer)
else:
    st.warning("Please upload a PDF and enter a Hugging Face API Key.")
