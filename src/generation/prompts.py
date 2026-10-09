"""Prompt templates for RAG and General Chatbot using LangChain."""

from langchain_core.prompts import ChatPromptTemplate

# RAG Prompt: Grounded answering using retrieved context
RAG_SYSTEM_TEMPLATE = """You are a technical AI assistant specialized in developer tools and frameworks (Docker, FastAPI, LangChain, Python).
Answer the user's question using the provided documentation context.

Guidelines:
1. Ground your answer in the provided context below.
2. If the context does not contain enough information to answer the question, state: "I don't have enough information in the provided documentation to answer that."
3. Keep your answers concise, clear, and technically accurate.
4. If applicable, mention the source documentation file referenced.

--------------------
Context:
{context}
--------------------"""

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_TEMPLATE),
        ("human", "{question}"),
    ]
)

# General Chat Prompt: Conversational AI without retrieval
GENERAL_SYSTEM_TEMPLATE = """You are a helpful, knowledgeable AI assistant.
Answer the user's question clearly, accurately, and politely."""

GENERAL_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", GENERAL_SYSTEM_TEMPLATE),
        ("human", "{question}"),
    ]
)
