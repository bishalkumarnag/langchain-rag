"""Interactive AI Chat Assistant supporting RAG and General queries using LangChain."""

import sys
from typing import Any, Dict, List, Optional, Union
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from src.generation.llm import get_chat_model
from src.generation.rag_chain import create_general_chain, create_rag_chain, format_docs
from src.retriever import RAGRetriever
from src.vectorstore import RAGVectorStore, load_chroma_vectorstore


class RAGChatAssistant:
    """Conversational Assistant capable of switching between RAG and General query modes."""

    def __init__(
        self,
        retriever: Optional[Union[BaseRetriever, VectorStore, RAGVectorStore]] = None,
        llm: Optional[BaseChatModel] = None,
        retriever_k: int = 3,
        search_type: str = "similarity",
    ):
        """Initialize the assistant.

        Args:
            retriever: LangChain BaseRetriever, VectorStore, or RAGVectorStore manager.
            llm: LangChain BaseChatModel instance. Auto-resolved from environment if None.
            retriever_k: Number of context documents to retrieve in RAG mode.
            search_type: 'similarity' or 'mmr'.
        """
        self.llm = llm or get_chat_model()
        self.retriever_k = retriever_k
        self.search_type = search_type

        # Resolve retriever
        if retriever is None:
            vectorstore = load_chroma_vectorstore(persist_directory="./chroma_db")
            self._retriever_manager = RAGRetriever(vectorstore=vectorstore)
        elif isinstance(retriever, RAGRetriever):
            self._retriever_manager = retriever
        elif isinstance(retriever, (VectorStore, RAGVectorStore)):
            self._retriever_manager = RAGRetriever(vectorstore=retriever)
        elif isinstance(retriever, BaseRetriever):
            self._base_retriever = retriever
            self._retriever_manager = None
        else:
            raise TypeError(f"Unsupported retriever type: {type(retriever)}")

        # Construct General Chain
        self._general_chain = create_general_chain(llm=self.llm)

    def _get_active_retriever(self, category: Optional[str] = None) -> BaseRetriever:
        """Resolve the active retriever based on search type and category filter."""
        if hasattr(self, "_base_retriever"):
            return self._base_retriever

        if category and category.lower() != "all":
            return self._retriever_manager.get_filtered_retriever(
                category=category.lower(),
                k=self.retriever_k,
            )

        if self.search_type == "mmr":
            return self._retriever_manager.get_mmr_retriever(k=self.retriever_k)

        return self._retriever_manager.get_similarity_retriever(k=self.retriever_k)

    def ask(
        self,
        query: str,
        use_rag: bool = True,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ask a question to the assistant in RAG mode or General mode.

        Args:
            query: The user question string.
            use_rag: If True, retrieves documentation context. If False, queries LLM directly.
            category: Optional documentation category to scope retrieval (e.g., 'docker', 'fastapi').

        Returns:
            Dict containing:
              - 'query': original question
              - 'answer': generated text response
              - 'mode': 'rag' or 'general'
              - 'sources': list of referenced document metadata dictionaries (if RAG)
        """
        if not use_rag:
            answer = self._general_chain.invoke({"question": query})
            return {
                "query": query,
                "answer": answer.strip(),
                "mode": "general",
                "sources": [],
            }

        # RAG Mode: Retrieve context and generate
        retriever = self._get_active_retriever(category=category)
        retrieved_docs: List[Document] = retriever.invoke(query)

        rag_chain = create_rag_chain(retriever=retriever, llm=self.llm)
        answer = rag_chain.invoke(query)

        sources = [
            {
                "file_name": doc.metadata.get("file_name", "Unknown"),
                "category": doc.metadata.get("category", "General"),
                "source": doc.metadata.get("source", ""),
                "start_index": doc.metadata.get("start_index", 0),
                "snippet": doc.page_content[:150].strip().replace("\n", " "),
            }
            for doc in retrieved_docs
        ]

        return {
            "query": query,
            "answer": answer.strip(),
            "mode": "rag",
            "sources": sources,
        }

    def interactive_chat(self) -> None:
        """Launch an interactive REPL chat session in the terminal."""
        print("\n" + "=" * 70)
        print(" 🤖 RAG AI ASSISTANT - INTERACTIVE CHAT MODE ".center(70, "="))
        print("=" * 70)
        print("Commands:")
        print("  • /rag             : Switch to RAG mode (answers from documentation)")
        print("  • /general         : Switch to General mode (direct LLM conversation)")
        print("  • /category <name> : Filter RAG to a category (docker, fastapi, langchain, python, all)")
        print("  • /exit or /quit   : Exit the assistant\n")

        current_mode = "rag"
        current_category: Optional[str] = None

        while True:
            try:
                scope_info = f" [Category: {current_category}]" if current_category else ""
                prompt_label = f"[{current_mode.upper()}{scope_info}] You: "
                user_input = input(prompt_label).strip()

                if not user_input:
                    continue

                if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                    print("\nGoodbye! 👋\n")
                    break

                elif user_input.lower() == "/rag":
                    current_mode = "rag"
                    print("--> Switched to RAG mode (grounded in documentation).\n")
                    continue

                elif user_input.lower() == "/general":
                    current_mode = "general"
                    print("--> Switched to General mode (general LLM queries).\n")
                    continue

                elif user_input.lower().startswith("/category"):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1 and parts[1].strip().lower() != "all":
                        current_category = parts[1].strip().lower()
                        print(f"--> Scoped RAG queries to category: '{current_category}'.\n")
                    else:
                        current_category = None
                        print("--> Category filter cleared (searching all documents).\n")
                    continue

                # Run query
                response = self.ask(
                    query=user_input,
                    use_rag=(current_mode == "rag"),
                    category=current_category,
                )

                print(f"\nAssistant: {response['answer']}\n")

                if response["sources"]:
                    print("📚 Sources Cited:")
                    for idx, src in enumerate(response["sources"], start=1):
                        print(f"   [{idx}] {src['file_name']} (Topic: {src['category']})")
                    print()

            except (KeyboardInterrupt, EOFError):
                print("\n\nSession terminated. Goodbye! 👋\n")
                break
            except Exception as e:
                print(f"\n⚠️ Error generating response: {e}\n")


def create_assistant(
    llm: Optional[BaseChatModel] = None,
    persist_directory: str = "./chroma_db",
) -> RAGChatAssistant:
    """Convenience factory returning a configured RAGChatAssistant instance."""
    store = load_chroma_vectorstore(persist_directory=persist_directory)
    return RAGChatAssistant(retriever=store, llm=llm)
