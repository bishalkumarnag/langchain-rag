"""Vector embeddings module using LangChain built-in embedding models.

This module strictly leverages:
- `langchain_core.embeddings.Embeddings`: LangChain's standard base abstraction.
- `langchain_google_genai.GoogleGenerativeAIEmbeddings`: Google Gemini embedding model.
- `langchain_openai.OpenAIEmbeddings`: OpenAI embedding model.
- `langchain_core.embeddings.FakeEmbeddings`: Built-in deterministic embeddings for offline testing.
"""

import os
from typing import List, Optional, Union
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings, FakeEmbeddings

# Load environment variables
load_dotenv()


def get_embeddings_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Embeddings:
    """Factory function returning a configured LangChain Embeddings instance.

    Args:
        provider: 'google', 'openai', or 'fake'. If None, auto-detected from environment.
        model: Model name/identifier. If None, uses provider default.

    Returns:
        Instance of langchain_core.embeddings.Embeddings.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    chosen_provider = (
        provider
        or os.getenv("EMBEDDING_PROVIDER")
        or os.getenv("LLM_PROVIDER")
        or ("google" if gemini_key else "openai" if openai_key else "fake")
    ).lower()

    if chosen_provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        model_name = model or os.getenv("GEMINI_EMBEDDING_MODEL") or "models/gemini-embedding-001"
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=gemini_key,
        )

    elif chosen_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        model_name = model or os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"
        return OpenAIEmbeddings(
            model=model_name,
            api_key=openai_key,
        )

    elif chosen_provider == "fake":
        return FakeEmbeddings(size=768)

    else:
        raise ValueError(
            f"Unsupported embedding provider: '{chosen_provider}'. "
            "Supported providers: 'google', 'openai', 'fake'."
        )


class RAGEmbeddings:
    """Wrapper managing vector embeddings generation using LangChain Embeddings."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        custom_embeddings: Optional[Embeddings] = None,
    ):
        """Initialize the embedding generator.

        Args:
            provider: 'google', 'openai', or 'fake'.
            model: Model identifier.
            custom_embeddings: Optional pre-configured LangChain Embeddings instance.
        """
        self.model = custom_embeddings or get_embeddings_model(provider=provider, model=model)
        self._dimension: Optional[int] = None

    @property
    def embeddings(self) -> Embeddings:
        """Return the underlying LangChain Embeddings object."""
        return self.model

    def embed_query(self, query: str) -> List[float]:
        """Generate a vector embedding for a user search query.

        Args:
            query: User search text string.

        Returns:
            List of floats representing the query embedding vector.
        """
        vector = self.model.embed_query(query)
        if self._dimension is None:
            self._dimension = len(vector)
        return vector

    def embed_documents(self, documents: Union[List[Document], List[str]]) -> List[List[float]]:
        """Generate vector embeddings for a list of LangChain Document chunks or raw texts.

        Args:
            documents: List of LangChain Document objects or list of text strings.

        Returns:
            List of embedding vectors (list of floats).
        """
        if not documents:
            return []

        # Extract text content if given LangChain Document objects
        if isinstance(documents[0], Document):
            texts = [doc.page_content for doc in documents]  # type: ignore
        else:
            texts = documents  # type: ignore

        vectors = self.model.embed_documents(texts)
        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])
        return vectors

    def get_dimension(self) -> int:
        """Return the vector dimensionality of this embedding model."""
        if self._dimension is None:
            sample_vector = self.embed_query("dimension probe")
            self._dimension = len(sample_vector)
        return self._dimension
