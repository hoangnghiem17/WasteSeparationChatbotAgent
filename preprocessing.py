from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
import re
import os
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

# Function to investigate the FAISS vector store
def investigate_vector_store(vector_store):
    # Check number of vectors in the index
    print(f"\nNumber of vectors in the FAISS index: {vector_store.index.ntotal}\n")

    # Access the internal dictionary of the docstore
    docs = vector_store.docstore._dict  # Accessing the private _dict attribute

    # Print the contents of the vector store
    print("Contents of the vector store:")
    for doc_id, document in docs.items():
        print(f"Document ID: {doc_id}")
        print(f"Content Preview: {document.page_content[:100]}...")  # Preview first 100 characters
        print(f"Metadata: {document.metadata}")
        print("-" * 40)

# Main script
if __name__ == "__main__":
    # Path to the uploaded PDF file
    file_path = "rag_docs/FES_waskommtwohinein.pdf"

    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(file_path)

    # Clean and normalize the extracted text
    cleaned_text = clean_text(extracted_text)

    # Chunk the cleaned text
    chunks = chunk_text(cleaned_text)

    # Print the chunks with clear separators
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:\n{'='*40}\n{chunk}\n")

    # Initialize the OpenAI embeddings model
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Create a FAISS vector store
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    # Print confirmation of storage
    print("\nVector store created with the following chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"Stored Chunk {i + 1}: {chunk[:100]}...\n")

    # Investigate the vector store
    investigate_vector_store(vector_store)
