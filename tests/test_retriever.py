"""Unit tests for LangChain Retriever module."""

import shutil
import tempfile
import unittest

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever

from src.vectorstore import RAGVectorStore
from src.retriever import RAGRetriever, create_retriever


class TestRetriever(unittest.TestCase):
    """Test suite for LangChain retriever strategies."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.embeddings = FakeEmbeddings(size=16)
        self.docs = [
            Document(
                page_content="Docker simplifies containerization for scalable microservices.",
                metadata={"source": "docker1.md", "category": "docker"},
            ),
            Document(
                page_content="Docker container images package dependencies securely.",
                metadata={"source": "docker2.md", "category": "docker"},
            ),
            Document(
                page_content="FastAPI dependency injection simplifies database sessions.",
                metadata={"source": "fastapi1.md", "category": "fastapi"},
            ),
            Document(
                page_content="Python object-oriented programming uses classes and methods.",
                metadata={"source": "python1.rst", "category": "python"},
            ),
        ]
        self.vectorstore_manager = RAGVectorStore(
            collection_name="test_retriever_collection",
            persist_directory=self.temp_dir,
            embeddings=self.embeddings,
        )
        self.vectorstore_manager.create_from_documents(self.docs)
        self.retriever_manager = RAGRetriever(vectorstore=self.vectorstore_manager)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_similarity_retriever(self):
        """Verify standard similarity retriever is a BaseRetriever and returns docs."""
        retriever = self.retriever_manager.get_similarity_retriever(k=2)
        self.assertIsInstance(retriever, BaseRetriever)
        self.assertIsInstance(retriever, VectorStoreRetriever)

        results = retriever.invoke("containerization and docker")
        self.assertEqual(len(results), 2)
        for doc in results:
            self.assertIsInstance(doc, Document)

    def test_mmr_retriever(self):
        """Verify Maximal Marginal Relevance (MMR) retriever."""
        retriever = self.retriever_manager.get_mmr_retriever(k=2, fetch_k=4, lambda_mult=0.5)
        self.assertIsInstance(retriever, BaseRetriever)

        results = retriever.invoke("container image packaging")
        self.assertGreaterEqual(len(results), 1)
        self.assertIsInstance(results[0], Document)

    def test_threshold_retriever(self):
        """Verify similarity score threshold retriever."""
        retriever = self.retriever_manager.get_threshold_retriever(score_threshold=-50.0, k=2)
        self.assertIsInstance(retriever, BaseRetriever)

        results = retriever.invoke("FastAPI dependency injection")
        self.assertIsInstance(results, list)

    def test_filtered_retriever(self):
        """Verify category-scoped retriever returns only matching documents."""
        retriever = self.retriever_manager.get_filtered_retriever(category="fastapi", k=2)
        results = retriever.invoke("application framework")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["category"], "fastapi")
        self.assertIn("FastAPI", results[0].page_content)

    def test_direct_retrieve_method(self):
        """Verify RAGRetriever.retrieve() convenience helper."""
        results = self.retriever_manager.retrieve("Python programming", k=1)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Document)


if __name__ == "__main__":
    unittest.main()
