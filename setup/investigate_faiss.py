
import os

from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def investigate_vector_store(faiss_store_path):
    """
    Investigates the contents of a FAISS vector store, printing out:
    - Total number of documents.
    - Number of documents (chunks) per source (from metadata).
    - Full content of the first document.

    Args:
        faiss_store_path (str): Path to the FAISS vector store.
    """
    # Load the FAISS vector store
    vector_store = FAISS.load_local(
        faiss_store_path,
        OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
        allow_dangerous_deserialization=True  # Ensure you trust the source of the data
    )

    # Determine the number of documents
    num_documents = len(vector_store.index_to_docstore_id)
    print(f"Total number of documents (chunks): {num_documents}")

    # Count the number of documents per source
    source_counts = {}
    for doc_id in vector_store.index_to_docstore_id.values():
        document = vector_store.docstore.search(doc_id)
        if isinstance(document, Document):
            source = document.metadata.get("source", "Unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        else:
            print(f"Document ID: {doc_id} not found in docstore.")

    # Print the counts per source
    print("\nNumber of documents per source:")
    for source, count in source_counts.items():
        print(f"{source}: {count}")

    # Display metadata and content for the first document
    first_document_found = False
    for doc_id in vector_store.index_to_docstore_id.values():
        document = vector_store.docstore.search(doc_id)
        if isinstance(document, Document):
            if not first_document_found:
                print("\nFirst Document Details:")
                print(f"Document ID: {doc_id}")
                print(f"Source: {document.metadata.get('source', 'Unknown')}")
                print(f"Full Content:\n{document.page_content}")
                first_document_found = True

# Directories and paths
faiss_store_path = "faiss_store"

# Create and investigate the FAISS vector store
investigate_vector_store(faiss_store_path)
