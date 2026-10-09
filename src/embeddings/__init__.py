"""Embeddings package providing vector embedding capabilities with LangChain."""

from src.embeddings.embedding_model import (
    RAGEmbeddings,
    get_embeddings_model,
)

__all__ = ["RAGEmbeddings", "get_embeddings_model"]
