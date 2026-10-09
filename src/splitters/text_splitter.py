"""Document splitting / chunking module using LangChain built-in splitters.

This module strictly leverages:
- `langchain_text_splitters.RecursiveCharacterTextSplitter`: Recursively splits by paragraphs,
  sentences, and characters to keep semantically related text together.
- `langchain_text_splitters.Language`: Language-specific separators (e.g., Markdown headers, code blocks).
- `langchain_core.documents.Document`: The standard LangChain Document abstraction.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


class RAGTextSplitter:
    """Document chunker using LangChain's built-in RecursiveCharacterTextSplitter."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        add_start_index: bool = True,
        strip_whitespace: bool = True,
    ):
        """Initialize the text splitter.

        Args:
            chunk_size: Maximum size of chunks (in characters).
            chunk_overlap: Number of characters to overlap between adjacent chunks.
            add_start_index: Whether to include 'start_index' in Document metadata.
            strip_whitespace: Whether to strip whitespace from start/end of each chunk.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_start_index = add_start_index
        self.strip_whitespace = strip_whitespace

        # Standard recursive text splitter
        self._default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=self.add_start_index,
            strip_whitespace=self.strip_whitespace,
        )

        # Markdown-aware recursive text splitter (preserves Markdown headers and code blocks)
        self._markdown_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=self.add_start_index,
            strip_whitespace=self.strip_whitespace,
        )

    def split_documents(
        self, documents: List[Document], language_aware: bool = True
    ) -> List[Document]:
        """Split a list of LangChain Document objects into smaller chunk Documents.

        Args:
            documents: List of LangChain Document objects to split.
            language_aware: If True, uses Markdown-optimized separators for .md/.mdx files.

        Returns:
            List of chunked LangChain Document objects.
        """
        chunked_documents: List[Document] = []

        for doc in documents:
            ext = doc.metadata.get("extension", "").lower()
            if language_aware and ext in (".md", ".mdx"):
                doc_chunks = self._markdown_splitter.split_documents([doc])
            else:
                doc_chunks = self._default_splitter.split_documents([doc])

            # Annotate each chunk with its index within the parent document
            for idx, chunk in enumerate(doc_chunks):
                chunk.metadata["chunk_index"] = idx
                chunk.metadata["total_chunks"] = len(doc_chunks)
                chunk.metadata["chunk_char_count"] = len(chunk.page_content)

            chunked_documents.extend(doc_chunks)

        return chunked_documents

    def split_text(self, text: str) -> List[str]:
        """Split raw text into smaller string chunks using LangChain's splitter.

        Args:
            text: Raw string content.

        Returns:
            List of string chunks.
        """
        return self._default_splitter.split_text(text)


def get_default_splitter(
    chunk_size: int = 1000, chunk_overlap: int = 200
) -> RecursiveCharacterTextSplitter:
    """Convenience factory returning LangChain's RecursiveCharacterTextSplitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )


def get_markdown_splitter(
    chunk_size: int = 1000, chunk_overlap: int = 200
) -> RecursiveCharacterTextSplitter:
    """Convenience factory returning LangChain's Markdown-aware RecursiveCharacterTextSplitter."""
    return RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    language_aware: bool = True,
) -> List[Document]:
    """Convenience function to split documents using LangChain's built-in splitter.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Character overlap between chunks.
        language_aware: Whether to use Markdown-aware separators for .md/.mdx files.

    Returns:
        List of chunked LangChain Document objects.
    """
    splitter = RAGTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents, language_aware=language_aware)
