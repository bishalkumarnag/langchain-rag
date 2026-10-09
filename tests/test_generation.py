"""Unit tests for LangChain Generation and RAGChatAssistant."""

import shutil
import tempfile
import unittest

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.generation import (
    RAGChatAssistant,
    create_general_chain,
    create_rag_chain,
    format_docs,
)
from src.generation.prompts import GENERAL_PROMPT_TEMPLATE, RAG_PROMPT_TEMPLATE
from src.vectorstore import RAGVectorStore


class TestGeneration(unittest.TestCase):
    """Test suite for LangChain RAG generation and interactive assistant."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.embeddings = FakeEmbeddings(size=16)
        self.docs = [
            Document(
                page_content="Docker containers provide isolated runtime environments.",
                metadata={"file_name": "docker-overview.md", "category": "docker"},
            ),
            Document(
                page_content="FastAPI allows declaring dependencies with Depends().",
                metadata={"file_name": "dependencies.md", "category": "fastapi"},
            ),
        ]
        self.store = RAGVectorStore(
            collection_name="test_gen_store",
            persist_directory=self.temp_dir,
            embeddings=self.embeddings,
        )
        self.store.create_from_documents(self.docs)

        # Mock LLM with dedicated responses
        self.mock_llm = FakeListChatModel(
            responses=[
                "Docker containers isolate applications from the host system.",
                "General response: Python is a multi-paradigm programming language.",
                "FastAPI response with dependencies.",
                "Additional mock response.",
            ]
        )
        self.assistant = RAGChatAssistant(
            retriever=self.store,
            llm=self.mock_llm,
            retriever_k=2,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_format_docs(self):
        """Verify format_docs produces formatted context with file citations."""
        formatted = format_docs(self.docs)
        self.assertIn("docker-overview.md", formatted)
        self.assertIn("dependencies.md", formatted)
        self.assertIn("Docker containers provide isolated", formatted)

    def test_rag_chain_execution(self):
        """Verify LCEL RAG chain runs and returns parsed text answer."""
        llm = FakeListChatModel(responses=["Docker containers isolate applications from the host."])
        retriever = self.store.as_retriever(k=1)
        rag_chain = create_rag_chain(retriever=retriever, llm=llm)
        result = rag_chain.invoke("What does Docker do?")
        self.assertIsInstance(result, str)
        self.assertIn("Docker", result)

    def test_general_chain_execution(self):
        """Verify general conversation chain runs without retrieval."""
        llm = FakeListChatModel(responses=["Python is a multi-paradigm programming language."])
        gen_chain = create_general_chain(llm=llm)
        result = gen_chain.invoke({"question": "Tell me about Python"})
        self.assertIsInstance(result, str)
        self.assertIn("Python", result)

    def test_assistant_rag_mode(self):
        """Verify assistant in RAG mode returns answer and cited sources."""
        llm = FakeListChatModel(responses=["Docker is a containerization platform."])
        assistant = RAGChatAssistant(retriever=self.store, llm=llm, retriever_k=2)
        response = assistant.ask("What is Docker?", use_rag=True)
        self.assertEqual(response["mode"], "rag")
        self.assertIsInstance(response["answer"], str)
        self.assertGreater(len(response["sources"]), 0)
        self.assertIn(response["sources"][0]["file_name"], ["docker-overview.md", "dependencies.md"])

    def test_assistant_general_mode(self):
        """Verify assistant in General mode returns direct answer with empty sources."""
        llm = FakeListChatModel(responses=["General conversation output."])
        assistant = RAGChatAssistant(retriever=self.store, llm=llm)
        response = assistant.ask("Hello!", use_rag=False)
        self.assertEqual(response["mode"], "general")
        self.assertIsInstance(response["answer"], str)
        self.assertEqual(len(response["sources"]), 0)


if __name__ == "__main__":
    unittest.main()
