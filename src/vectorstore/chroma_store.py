"""Vector store module using LangChain built-in Chroma integration.

This module strictly leverages:
- `langchain_chroma.Chroma`: LangChain's official Chroma vector store integration.
- `langchain_core.documents.Document`: Standard LangChain Document representation.
- `langchain_core.embeddings.Embeddings`: Base embedding abstraction.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from src.embeddings import get_embeddings_model

DEFAULT_COLLECTION_NAME = "rag_knowledge_base"
DEFAULT_PERSIST_DIR = "./chroma_db"


class RAGVectorStore:
    """Manager for Chroma vector store using official LangChain integration."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = DEFAULT_PERSIST_DIR,
        embeddings: Optional[Embeddings] = None,
    ):
        """Initialize the vector store.

        Args:
            collection_name: Name of the Chroma collection.
            persist_directory: Path to disk storage directory. If None, runs in-memory.
            embeddings: LangChain Embeddings instance. If None, auto-resolved.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embeddings = embeddings or get_embeddings_model()
        self._vectorstore: Optional[Chroma] = None

    @property
    def vectorstore(self) -> Chroma:
        """Return the underlying LangChain Chroma instance, initializing if needed."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
        return self._vectorstore

    def create_from_documents(
        self,
        documents: List[Document],
        batch_size: int = 40,
        force_reindex: bool = False,
    ) -> Chroma:
        """Index a list of LangChain Document chunks into Chroma.

        Args:
            documents: List of chunked Document objects.
            batch_size: Number of documents to index per batch.
            force_reindex: If False and documents are already indexed on disk, reuses existing store.

        Returns:
            The populated LangChain Chroma vector store.
        """
        if not documents:
            raise ValueError("Cannot initialize vector store with empty document list.")

        # Reuse existing persisted vector store if available and not forced to reindex
        if not force_reindex and self.persist_directory and Path(self.persist_directory).exists():
            current_count = self.count()
            if current_count > 0:
                return self.vectorstore

        # Clear existing store if forcing reindex
        if force_reindex:
            self.clear()

        # Index in safe batches to respect API provider rate limits
        first_batch = documents[:batch_size]
        self._vectorstore = Chroma.from_documents(
            documents=first_batch,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )

        # Add remaining documents in batches
        import time
        for i in range(batch_size, len(documents), batch_size):
            remaining_batch = documents[i : i + batch_size]
            time.sleep(1.0)
            self._vectorstore.add_documents(remaining_batch)

        return self._vectorstore

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add additional documents into the existing Chroma store.

        Args:
            documents: List of chunked Document objects.

        Returns:
            List of generated document IDs.
        """
        return self.vectorstore.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None,
    ) -> List[Document]:
        """Perform similarity search for relevant chunks against a user query.

        Args:
            query: User search text query.
            k: Number of most similar document chunks to return.
            filter: Optional metadata filter dictionary (e.g. {'category': 'docker'}).

        Returns:
            List of matching LangChain Document chunks.
        """
        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search returning documents alongside their distance score.

        Args:
            query: User search text query.
            k: Number of most similar document chunks to return.
            filter: Optional metadata filter dictionary.

        Returns:
            List of tuples of (Document, distance_score).
        """
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
        )

    def as_retriever(
        self,
        search_type: str = "similarity",
        k: int = 4,
        filter: Optional[Dict] = None,
        **extra_kwargs,
    ) -> VectorStoreRetriever:
        """Convert the Chroma vector store into a LangChain Retriever interface.

        Args:
            search_type: 'similarity', 'mmr', or 'similarity_score_threshold'.
            k: Number of documents to retrieve.
            filter: Metadata filter dictionary.
            **extra_kwargs: Additional parameters for the search (e.g. fetch_k, lambda_mult, score_threshold).

        Returns:
            LangChain VectorStoreRetriever instance.
        """
        search_kwargs: Dict = {"k": k, **extra_kwargs}
        if filter:
            search_kwargs["filter"] = filter

        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    def count(self) -> int:
        """Return the total number of vectors in this Chroma collection."""
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0

    def clear(self) -> None:
        """Delete and clear the stored vectors and persist directory."""
        if self._vectorstore is not None:
            try:
                self._vectorstore.delete_collection()
            except Exception:
                pass
            self._vectorstore = None

        if self.persist_directory and Path(self.persist_directory).exists():
            shutil.rmtree(self.persist_directory, ignore_errors=True)


def create_chroma_vectorstore(
    documents: List[Document],
    embeddings: Optional[Embeddings] = None,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    """Convenience factory creating and persisting a Chroma vector store."""
    manager = RAGVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embeddings=embeddings,
    )
    return manager.create_from_documents(documents)


def load_chroma_vectorstore(
    persist_directory: str = DEFAULT_PERSIST_DIR,
    embeddings: Optional[Embeddings] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    """Convenience factory loading an existing Chroma vector store from disk."""
    manager = RAGVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embeddings=embeddings,
    )
    return manager.vectorstore
