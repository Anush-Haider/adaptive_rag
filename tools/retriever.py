import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config.settings import settings


class VectorStoreManager:
    """
    Manages the initialization, ingestion, and retrieval interface
    for the local, open-source Chroma Vector Database.
    """

    def __init__(self):
        # 1. Initialize 100% local embedding model from HuggingFace
        print("--- INITIALIZING LOCAL EMBEDDING MODEL (all-MiniLM-L6-v2) ---")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.persist_directory = settings.CHROMA_PERSIST_DIR
        self._vectorstore = None

    def get_vectorstore(self, documents: List[Document] = None) -> Chroma:
        """
        Loads the existing local vector database or initializes a new one
        if documents are provided.
        """
        if self._vectorstore is not None:
            return self._vectorstore

        # Check if DB directory exists and is not empty
        db_exists = os.path.exists(self.persist_directory) and len(os.listdir(self.persist_directory)) > 0

        if db_exists:
            print(f"--- LOADING EXISTING CHROMA DB FROM {self.persist_directory} ---")
            self._vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            if not documents:
                raise ValueError(
                    f"No existing Chroma DB found at '{self.persist_directory}'. "
                    "You must provide a list of Documents to seed the database."
                )

            print(f"--- INITIALIZING NEW CHROMA DB AT {self.persist_directory} ---")
            self._vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )

        return self._vectorstore

    def get_retriever(self, documents: List[Document] = None):
        """
        Returns a standardized retriever interface configured with Similarity Search
        and Top-K filtering.
        """
        vdb = self.get_vectorstore(documents)
        # Using a Top-K search targeting the max results from our configurations
        return vdb.as_retriever(search_kwargs={"k": settings.MAX_SEARCH_RESULTS})


# Instantiate a global singleton worker for our graph nodes to tap into
vector_manager = VectorStoreManager()