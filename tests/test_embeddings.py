"""Unit tests for LangChain Vector Embeddings."""

import os
import unittest
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from src.embeddings import RAGEmbeddings, get_embeddings_model

load_dotenv()


class TestEmbeddings(unittest.TestCase):
    """Test suite for LangChain Embeddings integration."""

    def test_fake_embeddings_interface(self):
        """Verify LangChain FakeEmbeddings produces valid vector representations."""
        emb = RAGEmbeddings(provider="fake")
        self.assertIsInstance(emb.embeddings, Embeddings)
        self.assertEqual(emb.get_dimension(), 768)

        # Test single query embedding
        query_vec = emb.embed_query("What is Docker containerization?")
        self.assertIsInstance(query_vec, list)
        self.assertEqual(len(query_vec), 768)
        self.assertIsInstance(query_vec[0], float)

        # Test Document chunk embedding
        doc = Document(page_content="FastAPI is a modern web framework for Python.", metadata={"source": "test"})
        doc_vecs = emb.embed_documents([doc])
        self.assertEqual(len(doc_vecs), 1)
        self.assertEqual(len(doc_vecs[0]), 768)

    def test_factory_returns_embeddings_instance(self):
        """Verify factory returns proper LangChain Embeddings instance."""
        model = get_embeddings_model(provider="fake")
        self.assertIsInstance(model, FakeEmbeddings)

    def test_live_google_embeddings_if_key_present(self):
        """Verify live Google Gemini embeddings when API key is available."""
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            self.skipTest("GEMINI_API_KEY not configured, skipping live API test")

        emb = RAGEmbeddings(provider="google")
        query_vec = emb.embed_query("Explain LangChain RAG pipeline")
        self.assertIsInstance(query_vec, list)
        self.assertEqual(len(query_vec), 3072)
        self.assertEqual(emb.get_dimension(), 3072)

        # Test batch embedding
        doc = Document(page_content="Vector embeddings capture semantic meaning.", metadata={"source": "test"})
        doc_vecs = emb.embed_documents([doc])
        self.assertEqual(len(doc_vecs), 1)
        self.assertEqual(len(doc_vecs[0]), 3072)


if __name__ == "__main__":
    unittest.main()
