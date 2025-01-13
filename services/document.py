import os
import re
import logging

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text: str) -> str:
    """
    Cleans and normalizes text by replacing unwanted characters and normalizing spaces.
    
    Args:
        text (str): The input text to clean.
        
    Returns:
        str: The cleaned and normalized text.
    """
    replacements = {
        "": "-",  
        "": "-",  
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)
    
    return text

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file.
    
    Args:
        file_path (str): The path to the PDF file.
        
    Returns:
        str: The extracted text from all pages of the PDF file.
    """
    reader = PdfReader(file_path)
    
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
        
    return text

def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 140) -> list[str]:
    """
    Splits the input text into smaller chunks with optional overlap for efficient processing.

    Args:
        text (str): The text to be split into chunks.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 700.
        chunk_overlap (int, optional): The number of overlapping characters between chunks. Defaults to 140.

    Returns:
        list[str]: A list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", ".", " "]
    )
    
    return splitter.split_text(text)

def initialize_faiss(chunks: list[str]) -> FAISS:
    """
    Initializes a FAISS vector store with the provided text chunks.

    Args:
        chunks (list[str]): A list of text chunks to be embedded and stored in the vector store.

    Returns:
        FAISS: The initialized FAISS vector store containing embeddings of the text chunks.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    return FAISS.from_texts(chunks, embedding=embeddings)

import logging

def search_similar_chunks(query: str, vector_store: FAISS) -> list:
    """
    Searches the FAISS vector store for the most similar chunk to the input query.

    Args:
        query (str): The user query for similarity search.
        vector_store (FAISS): The initialized FAISS vector store.

    Returns:
        list: A list containing the best matching chunk (if found), represented as a document object.
    """
    try:
        # Perform similarity search with scores
        results_with_scores = vector_store.similarity_search_with_score(query, k=1)

        if results_with_scores:
            best_chunk, best_score = results_with_scores[0]

            # Log the similarity score of the best chunk
            logging.info(f"Best chunk similarity score: {best_score:.4f}")
            
            # Return the best chunk as a list
            return [best_chunk]
        else:
            # Log if no results were found
            logging.info("No similar chunks found.")
            return []

    except Exception as e:
        logging.error(f"Error during similarity search: {e}", exc_info=True)
        return []

def process_document(file_path: str, query: str, k: int = 2) -> list[str]:
    """
    Processes a document by extracting text, cleaning it, chunking it, embedding it, and performing a similarity search.

    Args:
        file_path (str): The path to the document to process.
        query (str): The query for which similar chunks are searched.
        k (int, optional): The number of most similar chunks to retrieve. Defaults to 3.

    Returns:
        list[str]: A list of the most similar text chunks.
    """
    # Extract, clean, and chunk text
    extracted_text = extract_text_from_pdf(file_path)
    cleaned_text = clean_text(extracted_text)
    chunks = chunk_text(cleaned_text)
    
    # Initialize vector store
    vector_store = initialize_faiss(chunks)
    
    # Perform similarity search
    similar_chunks = search_similar_chunks(query, vector_store)
    
    return [chunk.page_content for chunk in similar_chunks]
