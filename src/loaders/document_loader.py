"""Document loader module using LangChain built-in loaders.

This module leverages:
- `langchain_community.document_loaders.DirectoryLoader`: Loads entire directories recursively.
- `langchain_community.document_loaders.TextLoader`: Loads individual text, markdown, mdx, and rst files.
- `langchain_core.documents.Document`: The standard LangChain Document abstraction.
"""

from pathlib import Path
from typing import List, Optional
import warnings

# Suppress minor deprecation notices from langchain-community
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = ["**/*.md", "**/*.mdx", "**/*.rst"]


class RAGDocumentLoader:
    """Document loader using LangChain's built-in DirectoryLoader and TextLoader."""

    def __init__(
        self,
        base_dir: str = "data",
        glob_patterns: Optional[List[str]] = None,
        encoding: str = "utf-8",
    ):
        """Initialize the document loader.

        Args:
            base_dir: Root directory path where document files reside.
            glob_patterns: List of glob patterns to match. Defaults to ['.md', '.mdx', '.rst'].
            encoding: Encoding to use when loading files. Defaults to 'utf-8'.
        """
        self.base_dir = Path(base_dir)
        self.glob_patterns = glob_patterns or SUPPORTED_EXTENSIONS
        self.encoding = encoding

    def load_all(self) -> List[Document]:
        """Load all matching documents from the base directory using LangChain's DirectoryLoader.

        Returns:
            List of langchain_core.documents.Document objects.
        """
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory does not exist: {self.base_dir.resolve()}")

        loader = DirectoryLoader(
            path=str(self.base_dir),
            glob=self.glob_patterns,
            recursive=True,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": self.encoding},
        )
        documents = loader.load()
        self._enrich_metadata(documents)
        return documents

    def load_category(self, category: str) -> List[Document]:
        """Load documents from a specific subdirectory/category (e.g., 'docker', 'python').

        Args:
            category: Subfolder name within base_dir.

        Returns:
            List of langchain_core.documents.Document objects.
        """
        category_path = self.base_dir / category
        if not category_path.exists():
            raise FileNotFoundError(f"Category directory does not exist: {category_path.resolve()}")

        loader = DirectoryLoader(
            path=str(category_path),
            glob=self.glob_patterns,
            recursive=True,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": self.encoding},
        )
        documents = loader.load()
        self._enrich_metadata(documents)
        return documents

    def load_file(self, file_path: str) -> List[Document]:
        """Load an individual file using LangChain's built-in TextLoader.

        Args:
            file_path: Relative or absolute path to the file.

        Returns:
            List containing the loaded langchain_core.documents.Document.
        """
        target = Path(file_path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target.resolve()}")

        loader = TextLoader(file_path=str(target), encoding=self.encoding)
        documents = loader.load()
        self._enrich_metadata(documents)
        return documents

    def _enrich_metadata(self, documents: List[Document]) -> None:
        """Enrich document metadata with file details while keeping original source intact."""
        for doc in documents:
            source_path = Path(doc.metadata.get("source", ""))
            doc.metadata["file_name"] = source_path.name
            doc.metadata["extension"] = source_path.suffix
            # Determine category from parent directory if within base_dir
            try:
                rel_path = source_path.resolve().relative_to(self.base_dir.resolve())
                if len(rel_path.parts) > 1:
                    doc.metadata["category"] = rel_path.parts[0]
                else:
                    doc.metadata["category"] = "root"
            except (ValueError, Exception):
                doc.metadata["category"] = source_path.parent.name
            doc.metadata["char_count"] = len(doc.page_content)


def load_documents(data_path: str = "data") -> List[Document]:
    """Convenience function to load all documents from data directory.

    Args:
        data_path: Path to document data folder.

    Returns:
        List of langchain_core.documents.Document objects.
    """
    loader = RAGDocumentLoader(base_dir=data_path)
    return loader.load_all()


def load_single_document(file_path: str) -> List[Document]:
    """Convenience function to load a single document using LangChain TextLoader.

    Args:
        file_path: Path to target file.

    Returns:
        List of langchain_core.documents.Document objects.
    """
    loader = RAGDocumentLoader()
    return loader.load_file(file_path)
