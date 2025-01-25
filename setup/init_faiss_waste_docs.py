import os

from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path
import pytesseract
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

def split_document_by_section(file_path, output_dir, sections):
    """
    Splits a PDF document into multiple sections based on specified page ranges.

    Args:
        file_path (str): Path to the input PDF document.
        output_dir (str): Directory where the output PDFs will be saved.
        sections (dict): A dictionary where keys are section names, and values are lists of page ranges.
    """
    os.makedirs(output_dir, exist_ok=True)
    reader = PdfReader(file_path)

    for section_name, page_ranges in sections.items():
        writer = PdfWriter()
        for page_range in page_ranges:
            start_page, end_page = page_range
            for page_num in range(start_page - 1, end_page):
                writer.add_page(reader.pages[page_num])
        
        output_path = os.path.join(output_dir, f"{section_name}.pdf")
        with open(output_path, "wb") as output_file:
            writer.write(output_file)
        print(f"Created: {output_path}")

def extract_text_with_tesseract(input_dir, text_output_dir):
    """
    Extracts text from all PDFs in the input directory using Tesseract OCR 
    and writes the text to files in the output directory.

    Args:
        input_dir (str): Directory containing the PDFs to process.
        text_output_dir (str): Directory where the extracted text files will be saved.
    """
    os.makedirs(text_output_dir, exist_ok=True)

    for pdf_file in os.listdir(input_dir):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, pdf_file)
            text_output_path = os.path.join(text_output_dir, f"{os.path.splitext(pdf_file)[0]}.txt")

            print(f"Processing {pdf_file}...")

            images = convert_from_path(pdf_path, dpi=300)
            extracted_text = ""
            for page_num, image in enumerate(images):
                page_text = pytesseract.image_to_string(image, lang="deu")
                extracted_text += f"Page {page_num + 1}:\n{page_text}\n\n"

            with open(text_output_path, "w", encoding="utf-8") as text_file:
                text_file.write(extracted_text)
            print(f"Extracted text written to: {text_output_path}")

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
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

def create_faiss_vector_store(text_dir, faiss_store_path, chunk_size=700, chunk_overlap=140):
    """
    Creates a FAISS vector store from text files and saves it.

    Args:
        text_dir (str): Directory containing text files.
        faiss_store_path (str): Path to save the FAISS vector store.
        chunk_size (int): Maximum size of each chunk. Defaults to 700.
        chunk_overlap (int): Overlapping characters between chunks. Defaults to 140.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    documents = []
    metadata = []

    # Read each text file and prepare chunks with metadata
    for text_file in os.listdir(text_dir):
        if text_file.endswith(".txt"):
            file_path = os.path.join(text_dir, text_file)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                chunks = chunk_text(text, chunk_size, chunk_overlap)
                
                # Add each chunk and its metadata to the lists
                for chunk in chunks:
                    documents.append(chunk)
                    
                    # Remove the .txt extension from the file name for metadata
                    source_name = os.path.splitext(text_file)[0]
                    metadata.append({"source": source_name})  # Add cleaned document name as metadata

    # Create FAISS vector store
    vector_store = FAISS.from_texts(documents, embedding=embeddings, metadatas=metadata)
    vector_store.save_local(faiss_store_path)
    print(f"FAISS vector store saved at {faiss_store_path}")

# Define sections with non-contiguous page ranges
sections = {
    "general_waste": [(6, 8), (10, 10), (13, 13), (15, 15)],  
    "residual_waste": [(17, 17), (19, 21)],  
    "bio_waste": [(22, 23), (25, 26)], 
    "paper_waste": [(27, 31)],
    "glas_waste": [(32, 33)],
    "package_waste": [(34, 40)],
    "battery_waste": [(41, 46)],
    "electronic_waste": [(47, 52)],
    "other_waste": [(53, 64)]
}

# Paths
file_path = "rag_docs/2020_abfaelle_im_haushalt_bf.pdf"
pdf_output_dir = "rag_docs/doc_per_category_pdf"
text_output_dir = "rag_docs/doc_per_category_txt"
faiss_store_path = "faiss_store"

# Split the PDF into section-specific PDFs, extract text from PDF using OCR, chunk and embed the text in FAISS vector store
split_document_by_section(file_path, pdf_output_dir, sections)
extract_text_with_tesseract(pdf_output_dir, text_output_dir)
create_faiss_vector_store(text_output_dir, faiss_store_path)
