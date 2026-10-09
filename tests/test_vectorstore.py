"""Unit tests for LangChain Chroma Vector Store."""

import shutil
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

from src.vectorstore import RAGVectorStore


class TestChromaVectorStore(unittest.TestCase):
    """Test suite for Chroma vector store integration using LangChain."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.embeddings = FakeEmbeddings(size=16)
        self.sample_docs = [
            Document(
                page_content="Docker simplifies packaging and deploying applications in containers.",
                metadata={"source": "docker.md", "category": "docker", "chunk_index": 0},
            ),
            Document(
                page_content="FastAPI is a fast web framework for building APIs with Python.",
                metadata={"source": "fastapi.md", "category": "fastapi", "chunk_index": 0},
            ),
            Document(
                page_content="LangChain provides modular components for RAG and LLM applications.",
                metadata={"source": "langchain.md", "category": "langchain", "chunk_index": 0},
            ),
        ]
        self.store = RAGVectorStore(
            collection_name="test_collection",
            persist_directory=self.temp_dir,
            embeddings=self.embeddings,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_and_similarity_search(self):
        """Verify document indexing and similarity search."""
        self.store.create_from_documents(self.sample_docs)
        self.assertEqual(self.store.count(), 3)

        results = self.store.similarity_search("container deployment", k=1)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Document)
        self.assertIn("category", results[0].metadata)

    def test_similarity_search_with_score(self):
        """Verify similarity search returns document and numeric score."""
        self.store.create_from_documents(self.sample_docs)
        results = self.store.similarity_search_with_score("Python web API", k=2)
        self.assertEqual(len(results), 2)
        doc, score = results[0]
        self.assertIsInstance(doc, Document)
        self.assertIsInstance(score, float)

    def test_metadata_filtering(self):
        """Verify metadata filtering restricts search results."""
        self.store.create_from_documents(self.sample_docs)
        results = self.store.similarity_search(
            query="framework",
            k=3,
            filter={"category": "fastapi"},
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["category"], "fastapi")
        self.assertIn("FastAPI", results[0].page_content)

    def test_as_retriever(self):
        """Verify vectorstore transforms into a valid LangChain Retriever."""
        self.store.create_from_documents(self.sample_docs)
        retriever = self.store.as_retriever(k=2)
        self.assertIsInstance(retriever, VectorStoreRetriever)

        retrieved_docs = retriever.invoke("modular components")
        self.assertEqual(len(retrieved_docs), 2)
        self.assertIsInstance(retrieved_docs[0], Document)

    def test_persistence_and_reload(self):
        """Verify Chroma persists data to disk and reloads cleanly."""
        self.store.create_from_documents(self.sample_docs)

        # Create new instance pointing to same directory
        reloaded_store = RAGVectorStore(
            collection_name="test_collection",
            persist_directory=self.temp_dir,
            embeddings=self.embeddings,
        )
        self.assertEqual(reloaded_store.count(), 3)
        results = reloaded_store.similarity_search("Docker", k=1)
        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
