"""Main entry point for testing and demonstrating RAG Steps 1 through 6.

Pipeline Architecture:
  Step 1: Document Loading (DirectoryLoader, TextLoader)
  Step 2: Document Chunking (RecursiveCharacterTextSplitter)
  Step 3: Vector Embeddings (GoogleGenerativeAIEmbeddings / OpenAIEmbeddings)
  Step 4: Vector Store (Chroma with disk persistence)
  Step 5: Retriever (VectorStoreRetriever - Similarity, MMR, and Filtered)
  Step 6: Generation (Prompt Templates + Chat Model + RAG/General Assistant)

Usage:
  python main.py           # Runs the automated pipeline demonstration (Steps 1 to 6)
  python main.py --chat    # Launches interactive assistant chat session (RAG & General)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.loaders import RAGDocumentLoader
from src.splitters import RAGTextSplitter
from src.embeddings import RAGEmbeddings
from src.vectorstore import RAGVectorStore
from src.retriever import RAGRetriever
from src.generation import RAGChatAssistant
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


def print_divider(title: str = "") -> None:
    print("\n" + "=" * 70)
    if title:
        print(f" {title.upper()} ".center(70, "="))
        print("=" * 70)


def main():
    # Check if user requested interactive chat mode
    if len(sys.argv) > 1 and sys.argv[1] in ("--chat", "-c", "chat"):
        assistant = RAGChatAssistant()
        assistant.interactive_chat()
        return

    print_divider("RAG Chatbot Pipeline - Complete End-to-End Demo (Steps 1–6)")
    print("Strictly using LangChain built-in libraries:")
    print("  • Step 1 Loader:      DirectoryLoader, TextLoader (langchain_community)")
    print("  • Step 2 Splitter:    RecursiveCharacterTextSplitter (langchain_text_splitters)")
    print("  • Step 3 Embeddings:  GoogleGenerativeAIEmbeddings / OpenAIEmbeddings")
    print("  • Step 4 Vector Store: Chroma (langchain_chroma)")
    print("  • Step 5 Retriever:   VectorStoreRetriever (langchain_core.vectorstores)")
    print("  • Step 6 Generation:  ChatPromptTemplate, ChatGoogleGenerativeAI, LCEL Chain\n")

    # =========================================================================
    # STEP 1: DOCUMENT LOADING
    # =========================================================================
    print_divider("Step 1: Document Loading")
    loader = RAGDocumentLoader(base_dir="data")
    raw_documents = loader.load_all()
    print(f"Loaded {len(raw_documents)} raw documents from 'data/' directory.")

    categories = {}
    for doc in raw_documents:
        cat = doc.metadata.get("category", "unknown")
        categories.setdefault(cat, []).append(doc)

    print("\n" + "-" * 70)
    print(f"{'Category':<12} | {'Files':<6} | {'Total Characters':<18} | {'File Names'}")
    print("-" * 70)
    for cat, docs in sorted(categories.items()):
        total_chars = sum(len(d.page_content) for d in docs)
        file_names = ", ".join(d.metadata.get("file_name", "") for d in docs)
        print(f"{cat:<12} | {len(docs):<6} | {total_chars:<18,} | {file_names}")
    print("-" * 70)

    # =========================================================================
    # STEP 2: DOCUMENT CHUNKING / TEXT SPLITTING
    # =========================================================================
    print_divider("Step 2: Document Chunking (RecursiveCharacterTextSplitter)")
    chunk_size = 1000
    chunk_overlap = 200
    print(f"Configuration: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    splitter = RAGTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunked_documents = splitter.split_documents(raw_documents, language_aware=True)
    print(f"--> Transformed {len(raw_documents)} full documents into {len(chunked_documents)} chunks!\n")

    # =========================================================================
    # STEP 3: VECTOR EMBEDDINGS
    # =========================================================================
    print_divider("Step 3: Vector Embeddings (LangChain Embeddings)")
    rag_embeddings = RAGEmbeddings()
    underlying_model = rag_embeddings.embeddings
    dim = rag_embeddings.get_dimension()

    print(f"Embedding Class:   {type(underlying_model)}")
    print(f"Module:            {type(underlying_model).__module__}")
    print(f"Vector Dimension:  {dim} dimensions")

    # =========================================================================
    # STEP 4: VECTOR STORE (CHROMA)
    # =========================================================================
    print_divider("Step 4: Vector Store Indexing (LangChain Chroma)")
    chroma_manager = RAGVectorStore(
        collection_name="rag_knowledge_base",
        persist_directory="./chroma_db",
        embeddings=underlying_model,
    )

    initial_count = chroma_manager.count()
    if initial_count == 0:
        print("\nIndexing chunks into persistent Chroma vector store...")
        chroma_manager.create_from_documents(chunked_documents)
        print(f"✓ Successfully indexed {chroma_manager.count()} vectors into Chroma!")
    else:
        print(f"✓ Reusing existing persistent Chroma vector store with {initial_count} vectors.")

    # =========================================================================
    # STEP 5: DOCUMENT RETRIEVER (LANGCHAIN RETRIEVER)
    # =========================================================================
    print_divider("Step 5: Document Retriever (LangChain BaseRetriever)")
    retriever_manager = RAGRetriever(vectorstore=chroma_manager)
    sim_retriever = retriever_manager.get_similarity_retriever(k=2)
    mmr_retriever = retriever_manager.get_mmr_retriever(k=3, fetch_k=10, lambda_mult=0.5)

    print(f"Retriever Class:     {type(sim_retriever)}")
    print(f"Is BaseRetriever:    {isinstance(sim_retriever, BaseRetriever)}")
    print("✓ Configured Standard Similarity, MMR, and Category-Scoped Retrievers.")

    # =========================================================================
    # STEP 6: RESPONSE GENERATION (RAG & GENERAL CHAT ASSISTANT)
    # =========================================================================
    print_divider("Step 6: Response Generation (LangChain RAG Assistant)")
    assistant = RAGChatAssistant(retriever=chroma_manager, retriever_k=2)

    # 6A. RAG Query: Answering strictly from uploaded documentation
    rag_query = "What is Docker containerization and how does it isolate applications?"
    print(f"[Query 1 - RAG Mode]: \"{rag_query}\"")
    rag_response = assistant.ask(rag_query, use_rag=True)

    print("\n[Generated RAG Answer]:")
    print(rag_response["answer"])

    print("\n[Cited Sources from Knowledge Base]:")
    for idx, src in enumerate(rag_response["sources"], start=1):
        print(f"  [{idx}] {src['file_name']} (Topic: {src['category']}, Offset: {src['start_index']})")

    # 6B. Category-Scoped RAG Query: Scoped to FastAPI
    fastapi_query = "What is dependency injection and how do you use Depends()?"
    print(f"\n" + "-" * 70)
    print(f"[Query 2 - Scoped RAG Mode (category='fastapi')]: \"{fastapi_query}\"")
    fastapi_response = assistant.ask(fastapi_query, use_rag=True, category="fastapi")

    print("\n[Generated Scoped Answer]:")
    print(fastapi_response["answer"])

    print("\n[Cited Sources]:")
    for idx, src in enumerate(fastapi_response["sources"], start=1):
        print(f"  [{idx}] {src['file_name']} (Topic: {src['category']})")

    # 6C. General Query: Direct conversation with LLM without retrieval
    general_query = "In 2 sentences, what are the key benefits of writing modular Python code?"
    print(f"\n" + "-" * 70)
    print(f"[Query 3 - General Query Mode (No RAG)]: \"{general_query}\"")
    general_response = assistant.ask(general_query, use_rag=False)

    print("\n[Generated General Answer]:")
    print(general_response["answer"])
    print(f"Mode: {general_response['mode']} (Sources: {len(general_response['sources'])})")

    # Summary
    print_divider("Full RAG Pipeline Complete")
    print("✓ Step 1: Document Loader      -> DirectoryLoader & TextLoader")
    print("✓ Step 2: Document Chunker     -> RecursiveCharacterTextSplitter")
    print("✓ Step 3: Embeddings Model     -> GoogleGenerativeAIEmbeddings (3072 dims)")
    print("✓ Step 4: Chroma Vector DB     -> 80 vectors indexed & persistent")
    print("✓ Step 5: LangChain Retriever  -> Similarity & MMR retrieval strategies")
    print("✓ Step 6: Response Generation  -> LCEL RAG Chain + General Assistant Chat Model")
    print("\n💬 To launch the interactive CLI chat assistant, run:")
    print("   python main.py --chat")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
