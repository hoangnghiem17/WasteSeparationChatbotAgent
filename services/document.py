import os
import re

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Function to clean and normalize text
def clean_text(text):
    replacements = {
        "": "-",  # Replace bullet points
        "": "-",  # Replace other markers
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)  # Normalize spaces
    return text

# Function to extract text from the PDF file
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to chunk the extracted text
def chunk_text(text, chunk_size=700, chunk_overlap=140):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", ".", " "]
    )
    return splitter.split_text(text)

# Function to initialize FAISS vector store
def initialize_faiss(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.from_texts(chunks, embedding=embeddings)

# Function to search for similar chunks
def search_similar_chunks(query, vector_store, k=3):
    return vector_store.similarity_search(query, k=k)

# Main function to process document and return context
def process_document(file_path, query, k=3):
    # Extract, clean, and chunk text
    extracted_text = extract_text_from_pdf(file_path)
    cleaned_text = clean_text(extracted_text)
    chunks = chunk_text(cleaned_text)
    
    # Initialize vector store
    vector_store = initialize_faiss(chunks)
    
    # Perform similarity search
    similar_chunks = search_similar_chunks(query, vector_store, k)
    return [chunk.page_content for chunk in similar_chunks]
