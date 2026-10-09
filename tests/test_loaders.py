"""Unit tests for Document Loader and LangChain Document Objects."""

import unittest
from pathlib import Path
from langchain_core.documents import Document
from src.loaders import RAGDocumentLoader, load_documents, load_single_document


class TestDocumentLoader(unittest.TestCase):
    """Test suite for RAGDocumentLoader using LangChain built-ins."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = "data"
        cls.loader = RAGDocumentLoader(base_dir=cls.data_dir)

    def test_load_all_documents(self):
        """Verify that all documents in data/ are loaded as LangChain Document objects."""
        documents = self.loader.load_all()
        self.assertEqual(len(documents), 8, "Expected exactly 8 documents across all subfolders")

        for doc in documents:
            # Strictly verify it is an instance of LangChain's Document
            self.assertIsInstance(doc, Document, "Object must be an instance of langchain_core.documents.Document")
            # Verify page_content
            self.assertIsInstance(doc.page_content, str)
            self.assertGreater(len(doc.page_content.strip()), 0, "Document page_content should not be empty")
            # Verify metadata
            self.assertIn("source", doc.metadata)
            self.assertIn("category", doc.metadata)
            self.assertIn("file_name", doc.metadata)
            self.assertIn("extension", doc.metadata)

    def test_load_category(self):
        """Verify loading by category (e.g., langchain)."""
        langchain_docs = self.loader.load_category("langchain")
        self.assertEqual(len(langchain_docs), 2)
        for doc in langchain_docs:
            self.assertEqual(doc.metadata["category"], "langchain")
            self.assertTrue(doc.metadata["file_name"].endswith(".mdx"))

    def test_load_single_file(self):
        """Verify loading a single document using LangChain's TextLoader."""
        file_path = "data/fastapi/first-steps.md"
        docs = load_single_document(file_path)
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertIsInstance(doc, Document)
        self.assertEqual(doc.metadata["file_name"], "first-steps.md")
        self.assertIn("FastAPI", doc.page_content)


if __name__ == "__main__":
    unittest.main()
