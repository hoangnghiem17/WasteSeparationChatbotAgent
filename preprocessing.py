from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
import openai
import re
import os
from dotenv import load_dotenv

load_dotenv()

client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

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

# Function to search for similar chunks in the vector store
def search_similar_chunks(query, vector_store, k=3):
    return vector_store.similarity_search(query, k=k)

# Function to construct a prompt with context
def construct_prompt(query, context_chunks):
    context = "\n\n".join(context_chunks)
    return f"""
    You are an assistant specializing in answering questions based on the following context:
    
    {context}
    
    User's Question: {query}
    
    Provide a detailed and accurate answer using the context above.
    """

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

    # Initialize the OpenAI embeddings model
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Create a FAISS vector store
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    # Investigate the vector store
    print("Investigating the vector store...")
    print(f"Number of vectors: {vector_store.index.ntotal}")

    # User query
    user_query = "What should I do with old batteries?"

    # Perform similarity search
    print("\nPerforming similarity search...")
    similar_chunks = search_similar_chunks(user_query, vector_store)
    context_chunks = [chunk.page_content for chunk in similar_chunks]

    # Construct the prompt
    print("\nConstructing prompt...")
    prompt = construct_prompt(user_query, context_chunks)
    print(f"Generated Prompt:\n{prompt}\n")

    # Call OpenAI API with the constructed prompt
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.7
    )

    # Print the generated answer
    print("\nGenerated Answer:")
    print(response.choices[0].message.content)
