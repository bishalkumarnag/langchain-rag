"""Unit tests for Document Splitters and Chunking using LangChain."""

import unittest
from langchain_core.documents import Document
from src.loaders import load_documents, load_single_document
from src.splitters import RAGTextSplitter, split_documents, get_default_splitter


class TestTextSplitters(unittest.TestCase):
    """Test suite for LangChain RecursiveCharacterTextSplitter implementation."""

    @classmethod
    def setUpClass(cls):
        cls.raw_docs = load_documents("data")
        cls.splitter = RAGTextSplitter(chunk_size=1000, chunk_overlap=200)

    def test_split_documents_count_and_types(self):
        """Verify that splitting produces Document objects and increases count."""
        chunks = self.splitter.split_documents(self.raw_docs)
        self.assertGreater(len(chunks), len(self.raw_docs))
        self.assertGreater(len(chunks), 150)

        for chunk in chunks:
            self.assertIsInstance(chunk, Document)
            self.assertIsInstance(chunk.page_content, str)
            self.assertGreater(len(chunk.page_content.strip()), 0)
            # Metadata preservation and enrichment
            self.assertIn("source", chunk.metadata)
            self.assertIn("category", chunk.metadata)
            self.assertIn("start_index", chunk.metadata)
            self.assertIn("chunk_index", chunk.metadata)
            self.assertIn("total_chunks", chunk.metadata)

    def test_chunk_size_respects_limit(self):
        """Verify that chunks mostly adhere to maximum character limit."""
        chunks = self.splitter.split_documents(self.raw_docs)
        for chunk in chunks:
            # RecursiveCharacterTextSplitter splits by separator,
            # so chunk size shouldn't substantially exceed the target chunk_size (1000)
            self.assertLessEqual(len(chunk.page_content), 1200)

    def test_chunk_overlap_existence(self):
        """Verify that adjacent chunks from the same document exhibit overlap."""
        doc = load_single_document("data/fastapi/first-steps.md")
        chunks = self.splitter.split_documents(doc)
        self.assertGreater(len(chunks), 1)

        # Check overlap between first and second chunk
        chunk_0_tail = chunks[0].page_content[-50:].strip()
        self.assertIn(chunk_0_tail, chunks[1].page_content)

    def test_split_raw_text(self):
        """Verify splitting raw string text."""
        sample_text = ("Hello world! " * 200).strip()
        text_chunks = self.splitter.split_text(sample_text)
        self.assertIsInstance(text_chunks, list)
        self.assertGreater(len(text_chunks), 1)
        self.assertIsInstance(text_chunks[0], str)


if __name__ == "__main__":
    unittest.main()
