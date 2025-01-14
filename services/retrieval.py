import os
import logging

from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def retrieve_similar_chunks(query: str, faiss_store_path: str, rag_document: list[str], k: int = 3) -> list[str]:
    """
    Loads a FAISS vector store, filters by metadata, and performs a similarity search for the input query.

    Args:
        query (str): The user query for similarity search.
        faiss_store_path (str): Path to the saved FAISS vector store.
        rag_document (list[str]): List of relevant documents (sources) to filter by.
        k (int, optional): The number of most similar chunks to retrieve. Defaults to 3.

    Returns:
        list[str]: A list of the most similar text chunks.
    """
    try:
        logging.info(f"Starting similarity search for query: '{query}'")
        
        # Load the FAISS vector store
        vector_store = FAISS.load_local(
            faiss_store_path,
            OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
            allow_dangerous_deserialization=True  # Ensure you trust the source of the data
        )

        # Filter vector store documents that contain rag_document type
        filtered_docs = [
            doc for doc_id, doc in vector_store.docstore._dict.items()  # Access internal docstore (dictionary holding all documents)
            if doc.metadata.get("source") in rag_document
        ]
        if not filtered_docs:
            logging.warning(f"No matching documents found for sources: {rag_document}")
            return []

        logging.info(f"Filtered {len(filtered_docs)} documents for metadata sources: {rag_document}")

        # Debug filtered documents metadata
        for doc in filtered_docs:
            logging.debug(f"Filtered Document Metadata: {doc.metadata}")

        # Create a temporary vector store for the filtered documents
        embeddings = vector_store.embeddings  
        temp_vector_store = FAISS.from_documents(filtered_docs, embeddings)
        logging.info("Temporary vector store successfully created.")

        # Perform similarity search
        logging.info(f"Performing similarity search with top {k} results.")
        results_with_scores = temp_vector_store.similarity_search_with_score(query, k=k)

        if results_with_scores:
            for idx, (doc, score) in enumerate(results_with_scores, 1):
                logging.info(f"Chunk {idx} with score {score}: {doc.page_content[:100]})")
            return [result[0].page_content for result in results_with_scores]
        else:
            logging.info("No similar chunks found.")
            return []
    except Exception as e:
        logging.error(f"Error during similarity search: {e}", exc_info=True)
        return []

