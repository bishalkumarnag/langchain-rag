"""RAG generation chain built using LangChain Expression Language (LCEL).

This module strictly leverages:
- `langchain_core.runnables.RunnablePassthrough`: Passes the user query unchanged.
- `langchain_core.output_parsers.StrOutputParser`: Parses LLM output to clean string.
- `langchain_core.prompts.ChatPromptTemplate`: Parameterized chat prompt.
- `langchain_core.language_models.chat_models.BaseChatModel`: Chat model runnable.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough

from src.generation.llm import get_chat_model
from src.generation.prompts import GENERAL_PROMPT_TEMPLATE, RAG_PROMPT_TEMPLATE


def format_docs(docs: List[Document]) -> str:
    """Format retrieved Document chunks into a structured context string with source citations."""
    if not docs:
        return "No relevant documentation found."

    formatted_pieces = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("file_name") or doc.metadata.get("source", "Unknown")
        category = doc.metadata.get("category", "")
        header = f"--- Document Chunk {idx} (File: {source}" + (f", Topic: {category}" if category else "") + ") ---"
        formatted_pieces.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n".join(formatted_pieces)


def create_rag_chain(
    retriever: BaseRetriever,
    llm: Optional[BaseChatModel] = None,
    prompt: Optional[ChatPromptTemplate] = None,
) -> Runnable:
    """Construct a full RAG generation chain using LangChain Expression Language (LCEL).

    Pipeline Architecture:
      Question ──┬──> Retriever ──> format_docs ──> {context}  ──┐
                 └──> RunnablePassthrough ────────> {question} ─┴─> prompt ──> llm ──> StrOutputParser

    Args:
        retriever: LangChain BaseRetriever instance.
        llm: LangChain BaseChatModel instance (auto-resolved if None).
        prompt: LangChain ChatPromptTemplate instance (defaults to RAG_PROMPT_TEMPLATE).

    Returns:
        A LangChain Runnable ready to be executed with .invoke(question).
    """
    chat_model = llm or get_chat_model()
    prompt_template = prompt or RAG_PROMPT_TEMPLATE

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt_template
        | chat_model
        | StrOutputParser()
    )
    return rag_chain


def create_general_chain(
    llm: Optional[BaseChatModel] = None,
    prompt: Optional[ChatPromptTemplate] = None,
) -> Runnable:
    """Construct a direct general conversational chain without document retrieval.

    Pipeline:
      {"question": ...} ──> prompt ──> llm ──> StrOutputParser

    Args:
        llm: LangChain BaseChatModel instance.
        prompt: LangChain ChatPromptTemplate instance.

    Returns:
        A LangChain Runnable ready for invocation.
    """
    chat_model = llm or get_chat_model()
    prompt_template = prompt or GENERAL_PROMPT_TEMPLATE

    return prompt_template | chat_model | StrOutputParser()
