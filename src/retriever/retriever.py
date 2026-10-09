"""Retriever module using LangChain built-in retriever abstractions.

This module strictly leverages:
- `langchain_core.retrievers.BaseRetriever`: LangChain's standard runnable retriever interface.
- `langchain_core.vectorstores.VectorStoreRetriever`: The standard VectorStore-backed retriever.
- `langchain_core.documents.Document`: Standard LangChain Document representation.
"""

from typing import Dict, List, Optional, Union
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from src.vectorstore.chroma_store import RAGVectorStore


class RAGRetriever:
    """Manager configuring and creating LangChain VectorStore-backed retrievers."""

    def __init__(
        self,
        vectorstore: Union[VectorStore, RAGVectorStore],
    ):
        """Initialize with a LangChain VectorStore or RAGVectorStore manager.

        Args:
            vectorstore: Instance of VectorStore (e.g. Chroma) or RAGVectorStore.
        """
        if isinstance(vectorstore, RAGVectorStore):
            self._vectorstore = vectorstore.vectorstore
        else:
            self._vectorstore = vectorstore

    @property
    def vectorstore(self) -> VectorStore:
        """Return the underlying LangChain VectorStore."""
        return self._vectorstore

    def get_similarity_retriever(
        self,
        k: int = 4,
        filter: Optional[Dict] = None,
    ) -> VectorStoreRetriever:
        """Create a standard similarity search retriever.

        Args:
            k: Number of most similar document chunks to retrieve.
            filter: Optional metadata filter dictionary.

        Returns:
            LangChain VectorStoreRetriever instance.
        """
        search_kwargs: Dict = {"k": k}
        if filter:
            search_kwargs["filter"] = filter

        return self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

    def get_mmr_retriever(
        self,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict] = None,
    ) -> VectorStoreRetriever:
        """Create a Maximal Marginal Relevance (MMR) retriever.

        MMR balances similarity to the query with diversity among retrieved documents,
        preventing redundant or near-identical chunks from clogging the LLM context.

        Args:
            k: Number of final document chunks to return.
            fetch_k: Number of candidate documents to pass to MMR algorithm (fetch_k >= k).
            lambda_mult: 0 to 1 float. 0 maximizes diversity, 1 maximizes relevance. Default 0.5.
            filter: Optional metadata filter dictionary.

        Returns:
            LangChain VectorStoreRetriever instance with MMR strategy.
        """
        search_kwargs: Dict = {
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        }
        if filter:
            search_kwargs["filter"] = filter

        return self._vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
        )

    def get_threshold_retriever(
        self,
        score_threshold: float = 0.3,
        k: int = 4,
        filter: Optional[Dict] = None,
    ) -> VectorStoreRetriever:
        """Create a similarity score threshold retriever.

        Only retrieves document chunks whose similarity meets or exceeds the score threshold.

        Args:
            score_threshold: Minimum similarity threshold.
            k: Maximum number of documents to return.
            filter: Optional metadata filter dictionary.

        Returns:
            LangChain VectorStoreRetriever instance with threshold filtering.
        """
        search_kwargs: Dict = {
            "k": k,
            "score_threshold": score_threshold,
        }
        if filter:
            search_kwargs["filter"] = filter

        return self._vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=search_kwargs,
        )

    def get_filtered_retriever(
        self,
        category: str,
        k: int = 4,
    ) -> VectorStoreRetriever:
        """Create a retriever scoped strictly to a specific documentation category.

        Args:
            category: Subfolder/topic name (e.g., 'docker', 'fastapi', 'langchain', 'python').
            k: Number of documents to retrieve.

        Returns:
            LangChain VectorStoreRetriever instance scoped by metadata.
        """
        return self.get_similarity_retriever(
            k=k,
            filter={"category": category},
        )

    def retrieve(
        self,
        query: str,
        search_type: str = "similarity",
        k: int = 4,
        filter: Optional[Dict] = None,
        **extra_kwargs,
    ) -> List[Document]:
        """Convenience method to directly retrieve documents for a query.

        Args:
            query: User search query.
            search_type: 'similarity', 'mmr', or 'similarity_score_threshold'.
            k: Number of documents.
            filter: Optional metadata filter.
            **extra_kwargs: Extra search arguments (fetch_k, lambda_mult, score_threshold).

        Returns:
            List of retrieved LangChain Document objects.
        """
        search_kwargs = {"k": k, **extra_kwargs}
        if filter:
            search_kwargs["filter"] = filter

        retriever = self._vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
        return retriever.invoke(query)


def create_retriever(
    vectorstore: Union[VectorStore, RAGVectorStore],
    search_type: str = "similarity",
    k: int = 4,
    filter: Optional[Dict] = None,
    **extra_kwargs,
) -> VectorStoreRetriever:
    """Convenience factory creating a configured LangChain VectorStoreRetriever."""
    manager = RAGRetriever(vectorstore=vectorstore)
    if search_type == "mmr":
        return manager.get_mmr_retriever(k=k, filter=filter, **extra_kwargs)
    elif search_type == "similarity_score_threshold":
        return manager.get_threshold_retriever(k=k, filter=filter, **extra_kwargs)
    else:
        return manager.get_similarity_retriever(k=k, filter=filter)
