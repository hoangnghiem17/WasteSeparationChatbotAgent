import logging
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from agent.llm import openai_api_key

def retrieve_chunks(query: str, faiss_store_path: str, category: str, k: int = 2) -> List[str]:
    """
    Loads a FAISS vector store, filters by metadata and performs a similarity search on filtered documents.

    Args:
        query (str): The user query for similarity search.
        faiss_store_path (str): Path to the saved FAISS vector store.
        category (str): The waste category for filtering (e.g., "bio_waste").
        k (int, optional): The number of most similar chunks to retrieve. Defaults to 2.

    Returns:
        List[str]: A list of the most similar text chunks.
    """
    try:
        logging.info(f"Starting similarity search for query: '{query}' in category: '{category}'")

        # Load the FAISS vector store
        vector_store = FAISS.load_local(
            faiss_store_path,
            OpenAIEmbeddings(openai_api_key=openai_api_key),
            allow_dangerous_deserialization=True
        )

        # Filter vector store documents by the specified category
        filtered_docs = [
            doc for doc_id, doc in vector_store.docstore._dict.items()
            if doc.metadata.get("source") == category
        ]
        if not filtered_docs:
            logging.warning(f"No matching documents found for category: {category}")
            return []

        logging.info(f"Filtered {len(filtered_docs)} documents for category: {category}")

        # Create a temporary vector store for the filtered documents
        embeddings = vector_store.embeddings
        temp_vector_store = FAISS.from_documents(filtered_docs, embeddings)
        logging.info("Temporary vector store successfully created.")

        # Perform similarity search
        logging.info(f"Performing similarity search with top {k} results.")
        results_with_scores = temp_vector_store.similarity_search_with_score(query, k=k)

        if results_with_scores:
            for idx, (doc, score) in enumerate(results_with_scores, 1):
                logging.info(f"Chunk {idx} with score {score}.")
            return [result[0].page_content for result in results_with_scores]
        else:
            logging.info("No similar chunks found.")
            return []
    except Exception as e:
        logging.error(f"Error during similarity search: {e}")
        return []