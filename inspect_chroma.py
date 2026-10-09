"""Inspection script to view how data is structured and saved in Chroma."""

import json
from src.vectorstore import load_chroma_vectorstore


def inspect_chroma_db():
    print("=" * 70)
    print(" CHROMA VECTOR DATABASE - INTERNAL STORAGE & ATTRIBUTES INSPECTION ".center(70, "="))
    print("=" * 70)

    # 1. Load Chroma Vector Store
    vectorstore = load_chroma_vectorstore(persist_directory="./chroma_db")
    total_records = vectorstore._collection.count()

    print(f"\n📂 Persist Directory:  ./chroma_db")
    print(f"📦 Collection Name:    {vectorstore._collection.name}")
    print(f"📊 Total Records:       {total_records} indexed chunks\n")

    # 2. Fetch Sample Record from Chroma
    # We include all 4 primary components: documents, metadatas, embeddings, and IDs
    sample_data = vectorstore.get(
        limit=1,
        include=["documents", "metadatas", "embeddings"],
    )

    print("-" * 70)
    print("1. PRIMARY CHROMA ATTRIBUTES (COLUMNS)")
    print("-" * 70)
    print("Chroma stores data in 4 core columnar attributes:")
    print("  • 'ids'        : Unique identifier for each vector chunk (UUID)")
    print("  • 'documents'  : The raw text snippet (page_content)")
    print("  • 'metadatas'  : Key-value dictionary of metadata fields")
    print("  • 'embeddings' : High-dimensional float vector array\n")

    print("-" * 70)
    print("2. SAMPLE RECORD INSPECTION")
    print("-" * 70)

    # A. ID
    doc_id = sample_data["ids"][0]
    print(f"[Attribute: 'ids']\n  UUID: {doc_id}\n")

    # B. Metadata Fields
    metadata = sample_data["metadatas"][0]
    print("[Attribute: 'metadatas'] (Key-Value Schema):")
    for key, val in metadata.items():
        print(f"  • {key:<18} ({type(val).__name__:<4}): {val}")

    # C. Raw Document Text
    document_text = sample_data["documents"][0]
    print(f"\n[Attribute: 'documents'] (Raw Text Preview - first 250 characters):")
    preview = document_text[:250].strip().replace("\n", "\n    ")
    print(f"    \"{preview}...\"\n")

    # D. Embeddings Vector
    embedding_vector = sample_data["embeddings"][0]
    print(f"[Attribute: 'embeddings'] (Dense Vector):")
    print(f"  • Total Dimensions : {len(embedding_vector)} floats")
    print(f"  • Sample Values    : {embedding_vector[:5]} ...\n")

    print("-" * 70)
    print("3. PHYSICAL ON-DISK STORAGE LAYOUT (./chroma_db)")
    print("-" * 70)
    print("Chroma organizes persistence into two layers:")
    print("  1. chroma.sqlite3 : SQLite database storing metadata, text, IDs, and config.")
    print("  2. <uuid_folder>/ : HNSW binary index files (data_level0.bin, header.bin)")
    print("                      for lightning-fast approximate nearest neighbor vector search.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    inspect_chroma_db()
