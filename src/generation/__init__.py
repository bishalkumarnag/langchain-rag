"""Generation package providing LLM response generation and interactive chat with LangChain."""

from src.generation.assistant import (
    RAGChatAssistant,
    create_assistant,
)
from src.generation.llm import get_chat_model
from src.generation.prompts import (
    GENERAL_PROMPT_TEMPLATE,
    RAG_PROMPT_TEMPLATE,
)
from src.generation.rag_chain import (
    create_general_chain,
    create_rag_chain,
    format_docs,
)

__all__ = [
    "RAGChatAssistant",
    "create_assistant",
    "get_chat_model",
    "RAG_PROMPT_TEMPLATE",
    "GENERAL_PROMPT_TEMPLATE",
    "create_rag_chain",
    "create_general_chain",
    "format_docs",
]
