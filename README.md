# Modular RAG Chatbot (LangChain)

A modular Retrieval-Augmented Generation (RAG) chatbot implementation built strictly with official **LangChain** built-in components.

---

## 🏗️ Project Architecture

The project follows a clean, decoupled modular pipeline as defined below:

```text
langchain-rag/
├── data/
│   ├── langchain/
│   │   ├── providers-and-models.mdx
│   │   └── learn.mdx
│   ├── fastapi/
│   │   ├── first-steps.md
│   │   └── dependencies.md
│   ├── docker/
│   │   ├── docker-overview.md
│   │   └── what-is-a-container.md
│   └── python/
│       ├── controlflow.rst
│       └── classes.rst
├── src/
│   ├── loaders/        <-- [COMPLETED: STEP 1] Document Loader & LangChain Documents
│   ├── splitters/      <-- [COMPLETED: STEP 2] Text Chunking & Splitting
│   ├── embeddings/     <-- [COMPLETED: STEP 3] Vector Embeddings
│   ├── vectorstore/    <-- [COMPLETED: STEP 4] Vector Store Indexing (Chroma)
│   ├── retriever/      <-- [COMPLETED: STEP 5] Document Retrieval
│   └── generation/     <-- [COMPLETED: STEP 6] LLM Response Generation & Interactive Chatbot
├── tests/
│   ├── test_loaders.py
│   ├── test_splitters.py
│   ├── test_embeddings.py
│   ├── test_vectorstore.py
│   ├── test_retriever.py
│   └── test_generation.py
├── main.py
├── inspect_chroma.py
├── requirements.txt
└── README.md
```

---

## 📌 Step 1: Document Loader & LangChain Document Objects

In this phase, we implement document ingestion using **strictly in-built LangChain components**:

### 💡 Mental Model: Think of the Loader as an Adapter

This is probably the best mental model.

```text
               Different Sources
           ┌──────────┼──────────┐
          PDF        HTML       CSV
           │          │          │
           ▼          ▼          ▼
       PDF Loader Web Loader CSV Loader
           └──────────┼──────────┘
                      │
                      ▼
              LangChain Document
```

Everything gets converted into a common interface:

```python
Document(
    page_content="Docker is an open platform for developing, shipping, and running applications...",
    metadata={
        "source": "data/docker/docker-overview.md",  # Compulsory: Path to source file
        "category": "docker",                        # Category / directory name
        "file_name": "docker-overview.md",           # File name
        "extension": ".md",                          # File type
        "char_count": 9938                           # Total document character count
    }
)
```

---

### Key LangChain Components Used:
1. **`DirectoryLoader`** (`langchain_community.document_loaders.DirectoryLoader`):
   - Recursively traverses directories and subdirectories (`docker/`, `fastapi/`, `langchain/`, `python/`).
   - Uses glob filters (`*.md`, `*.mdx`, `*.rst`) to match all target files.
   - Automatically handles non-hidden files without custom parsing logic.
2. **`TextLoader`** (`langchain_community.document_loaders.TextLoader`):
   - In-built text loader passed as `loader_cls` to read raw file contents with proper UTF-8 encoding.
3. **`Document`** (`langchain_core.documents.Document`):
   - The fundamental standard abstraction in LangChain.
   - **`page_content`**: The raw text content of the document.
   - **`metadata`**: Key-value dictionary containing origin metadata (`source`, `category`, `file_name`, `extension`, `char_count`).

---

## 📌 Step 2: Document Chunking & Text Splitters

In this phase, we transform full documents into smaller, semantically coherent chunks using **strictly in-built LangChain splitters**.

### 💡 Why Chunking is Critical in RAG:
1. **Embedding Quality**: Embedding models have input token limits (e.g. 512–8192 tokens) and perform best when text focuses on a single concept rather than an entire multi-page document.
2. **Retrieval Precision**: Retrieving a targeted 500-1000 character chunk gives the LLM high signal-to-noise ratio, minimizing irrelevant context and hallucinations.
3. **Context Window Efficiency**: Enables fitting multiple diverse snippets from different files into a prompt.

### 💡 Visual Mental Model: Chunking with Overlap

```text
Document: [==========================================================]

Chunk 1:  [======================]
                 (Overlap: 200 chars)
Chunk 2:              [======================]
                             (Overlap: 200 chars)
Chunk 3:                          [======================]
```

> **Why Chunk Overlap matters**: Overlap ensures sentences and thoughts cut across boundaries are not lost during similarity search.

### Key LangChain Components Used:
1. **`RecursiveCharacterTextSplitter`** (`langchain_text_splitters.RecursiveCharacterTextSplitter`):
   - Recursively tries splitting by `["\n\n", "\n", " ", ""]` to keep paragraphs, sentences, and words together.
   - Configured with `chunk_size=1000` and `chunk_overlap=200`.
   - `add_start_index=True`: Captures the exact character position (`start_index`) inside the source file.
2. **`Language.MARKDOWN`** (`langchain_text_splitters.Language`):
   - Markdown-aware recursive splitting that respects Markdown headers (`#`, `##`) and code fences (````) so code snippets and sections aren't chopped mid-syntax.
3. **Chunk Metadata Enrichment**:
   - Preserves all parent document metadata (`source`, `category`, `file_name`, etc.).
   - Adds chunk tracking metadata: `chunk_index`, `total_chunks`, and `chunk_char_count`.

### 📄 What a Chunk Document Object Looks Like:

After splitting, each chunk is encapsulated in a LangChain `Document` object with its text snippet and enriched metadata:

```python
Document(
    page_content="Docker enables you to separate your applications from your infrastructure so you can deliver software quickly...",
    metadata={
        # Inherited from parent document (Compulsory origin tracking):
        "source": "data/docker/docker-overview.md",  # Path to original source file
        "category": "docker",                        # Topic / documentation category
        "file_name": "docker-overview.md",           # Original file name
        "extension": ".md",                          # File format extension

        # Generated by LangChain text splitter:
        "start_index": 677,                          # Compulsory: Character offset in source file

        # Chunk tracking metadata:
        "chunk_index": 1,                            # Sequential index of this chunk (0-based)
        "total_chunks": 14,                          # Total number of chunks for this file
        "chunk_char_count": 468                      # Character length of this chunk
    }
)
```

---

## 📌 Step 3: Vector Embeddings

In this phase, we convert chunked text into dense mathematical vectors (embeddings) using **strictly in-built LangChain embedding models**.

### 💡 Visual Mental Model: The Semantic Vector Space

Embedding models transform human language into high-dimensional vectors where semantically similar meanings land close together:

```text
Text Chunk / Query                   Embedding Model                  Vector Representation
---------------------------------------------------------------------------------------------
"Docker containers isolate apps" ---> [ LangChain Embeddings ] ---> [-0.0042, 0.0104, -0.0567, ...] (3072 dims)
"Container virtualization engine" ---> [ LangChain Embeddings ] ---> [-0.0039, 0.0098, -0.0521, ...] (High Similarity)
"Python class inheritance object" ---> [ LangChain Embeddings ] ---> [-0.0152, 0.0080, -0.0746, ...] (Low Similarity)
```

### Key LangChain Components Used:
1. **`Embeddings`** (`langchain_core.embeddings.Embeddings`):
   - The universal base class defining the contract:
     - `embed_documents(texts: List[str]) -> List[List[float]]`
     - `embed_query(text: str) -> List[float]`
2. **`GoogleGenerativeAIEmbeddings`** (`langchain_google_genai.GoogleGenerativeAIEmbeddings`):
   - Official integration with Google Gemini embedding models (`models/gemini-embedding-001`, 3072 dimensions).
3. **`OpenAIEmbeddings`** (`langchain_openai.OpenAIEmbeddings`):
   - Official integration with OpenAI embedding models (`text-embedding-3-small` / `text-embedding-ada-002`).
4. **`FakeEmbeddings`** (`langchain_core.embeddings.FakeEmbeddings`):
   - Built-in deterministic mock embeddings for offline unit tests and continuous integration without API keys.

### 📄 What an Embedded Chunk Vector Looks Like:

```python
# Query or Chunk Text:
"How do Docker containers work and isolate applications?"

# LangChain Vector Output:
[
    -0.004206788,
    -0.0048801424,
     0.010426093,
    -0.05671857,
     0.00376348,
    ... # 3,072 float dimensions
]
```

---

## 📌 Step 4: Vector Store Indexing (Chroma)

In this phase, we index chunk embeddings and metadata into **Chroma**, an open-source vector database, strictly using LangChain's official `langchain_chroma` integration.

### 💡 Visual Mental Model: The Vector Database

The vector database acts as a specialized spatial index (using Hierarchical Navigable Small World graphs / HNSW) for searching high-dimensional vectors:

```text
Indexing Phase:
  Document Chunks ---> Embedding Model ---> [ Chroma Vector Database ]
  (Text + Metadata)        (Dense Vectors)      ├── Vectors (HNSW Graph)
                                                ├── Metadata (Source, Category, Index)
                                                └── Disk Persistence (./chroma_db)

Search Phase:
  User Query      ---> Embedding Model ---> Cosine/L2 Search in Chroma ---> Top K Document Chunks
  "Docker tools"       (Query Vector)                                       (Ranked with Distance)
```

### Key LangChain Components Used:
1. **`Chroma`** (`langchain_chroma.Chroma`):
   - LangChain's official vector store class wrapping ChromaDB.
   - Built-in methods:
     - `Chroma.from_documents(documents, embedding, persist_directory=...)`
     - `vectorstore.similarity_search(query, k=...)`
     - `vectorstore.similarity_search_with_score(query, k=...)`
     - `vectorstore.as_retriever(search_kwargs=...)`
2. **Disk Persistence**:
   - Stores vectors and metadata locally under `./chroma_db` across sessions without re-embedding on every run.
3. **Metadata Filtering**:
   - Enables scoping similarity searches by metadata attributes (e.g. `filter={"category": "fastapi"}`).

### 📄 What a Chroma Vector Search Result Looks Like:

```python
# Query: "How do Docker containers work and isolate applications?"
# Returns tuples of (Document, distance_score):
[
    (
        Document(
            page_content="## The Docker platform\nDocker provides the ability to package and run an application...",
            metadata={
                "source": "data/docker/docker-overview.md",
                "category": "docker",
                "file_name": "docker-overview.md",
                "start_index": 0,
                "chunk_index": 0
            }
        ),
        0.5058  # L2 / Cosine Distance (lower is closer)
    )
]
```

### 🔍 Inspecting Chroma Database Internals:

To view how data, column names, and metadata are physically stored in Chroma:

```bash
python inspect_chroma.py
```

Chroma stores each record under 4 main attributes:
- **`ids`**: Unique UUID assigned to the chunk.
- **`documents`**: The raw text content of the chunk (`page_content`).
- **`metadatas`**: Key-value JSON dictionary (`source`, `category`, `file_name`, `start_index`, `chunk_index`, etc.).
- **`embeddings`**: The 3,072-dimensional float vector array.

---

## 📌 Step 5: Document Retriever (LangChain BaseRetriever)

In this phase, we convert the vector database into a LangChain **Retriever**—a standard Runnable component that accepts a string query and returns relevant `Document` chunks.

### 💡 Visual Mental Model: VectorStore vs. Retriever

| Concept | Vector Store (Chroma) | Retriever (`VectorStoreRetriever`) |
| :--- | :--- | :--- |
| **Role** | Storage & Indexing Engine (HNSW graph) | Retrieval Execution Component (LCEL Runnable) |
| **Input** | Vector embeddings, Documents, IDs | Natural Language Query (`str`) |
| **Output** | Stored IDs, raw vectors, distance floats | List of LangChain `Document` objects |
| **Pipeline Position** | Underlying Database Infrastructure | Piped directly into Prompt / LLM (`retriever \| prompt \| llm`) |

```text
User Query: "How does Docker isolate apps?"
     │
     ▼
[ LangChain Retriever (Runnable) ] ─── (implements .invoke())
     ├── Strategy A: Standard Similarity (k=4 nearest neighbors)
     ├── Strategy B: MMR (Maximal Marginal Relevance - balances relevance + diversity)
     ├── Strategy C: Score Threshold (filters out low-confidence chunks)
     └── Strategy D: Metadata Filter (scopes search to specific category)
     │
     ▼
[ List of Relevant LangChain Document Chunks ]
```

### Key LangChain Components Used:
1. **`BaseRetriever`** (`langchain_core.retrievers.BaseRetriever`):
   - The abstract base class powering all LangChain retrievers, exposing `.invoke(query)` and `.ainvoke(query)`.
2. **`VectorStoreRetriever`** (`langchain_core.vectorstores.base.VectorStoreRetriever`):
   - Vector store-backed retriever generated via `vectorstore.as_retriever()`.
3. **Retrieval Strategies**:
   - **`similarity`**: Standard cosine/distance nearest neighbors.
   - **`mmr` (Maximal Marginal Relevance)**: Fetches `fetch_k` candidate documents, then optimizes for both relevance to the query and diversity among the results, preventing repetitive redundant chunks.
   - **`similarity_score_threshold`**: Sets a minimum quality cutoff score.
   - **Metadata Filtering**: Scopes queries (e.g. `filter={"category": "fastapi"}`).

### 📄 What a Retriever Invocation Looks Like:

```python
# Create MMR retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.5}
)

# Standard LangChain Runnable invocation:
docs = retriever.invoke("What is Docker containerization?")
# Returns: [Document(page_content="...", metadata={...}), ...]
```

---

## 📌 Step 6: Response Generation & Interactive AI Chat Assistant

In this final phase, we connect the entire pipeline from query to generated answer using **strictly in-built LangChain components** and **LangChain Expression Language (LCEL)**.

### 💡 Mental Model: The LCEL Composition Pipeline

Rather than imperatively writing loops and manual string replacements, LangChain connects components using the Unix-style pipe operator (`|`):

```text
                               ┌────────────────────────────────┐
                               │     User Query: "..."         │
                               └───────────────┬────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
         ┌───────────────────────────┐                   ┌───────────────────────────┐
         │ LangChain BaseRetriever   │                   │    RunnablePassthrough    │
         │  (Chroma HNSW Similarity) │                   │  (Keeps Original Question)│
         └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                       │                                               │
                       ▼                                               │
         ┌───────────────────────────┐                                 │
         │   format_docs() Helper    │                                 │
         │ (Formats Chunks + Sources)│                                 │
         └─────────────┬─────────────┘                                 │
                       │                                               │
                       └───────────────────────┬───────────────────────┘
                                               ▼
                                 ┌───────────────────────────┐
                                 │    ChatPromptTemplate     │
                                 │   System: Grounded Prompt │
                                 │   Human: {question}       │
                                 └─────────────┬─────────────┘
                                               ▼
                                 ┌───────────────────────────┐
                                 │       BaseChatModel       │
                                 │ (Gemini 3.5 / GPT-4o-mini)│
                                 └─────────────┬─────────────┘
                                               ▼
                                 ┌───────────────────────────┐
                                 │      StrOutputParser      │
                                 │ (Parses AIMessage to str) │
                                 └─────────────┬─────────────┘
                                               ▼
                                 ┌───────────────────────────┐
                                 │     Final Response &      │
                                 │     Source Citations      │
                                 └───────────────────────────┘
```

### Key LangChain Components Used:
1. **`ChatPromptTemplate`** (`langchain_core.prompts`):
   - Parameterized prompt template enforcing strict grounding in retrieved documentation and graceful fallback when context is absent.
2. **`BaseChatModel`** (`langchain_google_genai.ChatGoogleGenerativeAI` / `langchain_openai.ChatOpenAI`):
   - Native LangChain chat models providing streaming, async, and structured execution.
3. **`StrOutputParser`** (`langchain_core.output_parsers.StrOutputParser`):
   - Extracts the clean markdown/text string from the model's `AIMessage` output.
4. **`RunnablePassthrough`** (`langchain_core.runnables.RunnablePassthrough`):
   - In-built LangChain primitive forwarding the input query directly into the prompt without alteration.
5. **`RAGChatAssistant`** (`src.generation.assistant.py`):
   - An interactive conversational assistant capable of switching seamlessly between **RAG Mode** (grounded in indexed docs with file citations), **Scoped RAG Mode** (filtered by topic), and **General Mode** (pure conversational AI).

### 📄 What the LCEL Chain Construction Looks Like:

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.generation.prompts import RAG_PROMPT_TEMPLATE
from src.generation.llm import get_chat_model

# Construct the end-to-end LCEL chain:
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | RAG_PROMPT_TEMPLATE
    | get_chat_model()
    | StrOutputParser()
)

# Invoke the entire chain in a single call:
answer = rag_chain.invoke("What is a Docker container?")
```

---

## 🚀 Setup Instructions

### 1. Prerequisites
- Python 3.10+ (Tested with Python 3.14)
- Terminal / CLI

### 2. Create and Activate Virtual Environment

```bash
# Navigate to project folder
cd "langchain-rag"

# Create a virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies

Install the official LangChain packages specified in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project root:

```bash
# For Google Gemini (recommended - free tier available):
GEMINI_API_KEY="your-gemini-api-key"
GEMINI_DEFAULT_MODEL="gemini-3.5-flash-lite"

# Or for OpenAI:
OPENAI_API_KEY="your-openai-api-key"
```

---

## 🧪 How to Run and How to Test

### Option A: Run the Full Pipeline Demonstration (`main.py`)

Run `main.py` to see the complete end-to-end execution of all 6 steps:

```bash
python main.py
```

#### What It Demonstrates:
1. **Step 1 (Loader)**: Ingests 8 documents across `docker`, `fastapi`, `langchain`, `python`.
2. **Step 2 (Chunker)**: Splits documents into clean markdown/rst chunks using `RecursiveCharacterTextSplitter`.
3. **Step 3 (Embeddings)**: Embeds chunks using LangChain embeddings (Gemini 3072 dims / OpenAI).
4. **Step 4 (Vector DB)**: Persists and indexes chunks in Chroma vector store.
5. **Step 5 (Retriever)**: Executes Similarity and MMR retrieval.
6. **Step 6 (Generation)**:
   - **Query 1 (Full RAG Mode)**: Answers technical question grounded in documentation with source citations.
   - **Query 2 (Scoped RAG Mode)**: Answers a category-constrained question (`category="fastapi"`).
   - **Query 3 (General Query Mode)**: Direct conversational query without retrieval.

---

### Option B: Interactive Terminal Chat Assistant (`main.py --chat`)

Launch an interactive chat session to query the assistant dynamically:

```bash
python main.py --chat
```

#### Assistant Commands:
- `/rag` — Switch to RAG mode (grounded in documentation, cites sources).
- `/general` — Switch to General conversation mode (no retrieval).
- `/category <name>` — Scope RAG to a specific documentation category (`docker`, `fastapi`, `langchain`, `python`, or `all`).
- `/exit` or `/quit` — Exit the assistant.

#### Interactive CLI Session Example:
```text
======================================================================
============ 🤖 RAG AI ASSISTANT - INTERACTIVE CHAT MODE =============
======================================================================
Commands:
  • /rag             : Switch to RAG mode (answers from documentation)
  • /general         : Switch to General mode (direct LLM conversation)
  • /category <name> : Filter RAG to a category (docker, fastapi, langchain, python, all)
  • /exit or /quit   : Exit the assistant

[RAG] You: What is a Docker container?

Assistant: A Docker container is a runnable instance of an image. It is a lightweight,
standalone, and executable package that contains everything needed to run an application,
including code, runtime, system tools, and libraries.

📚 Sources Cited:
   [1] what-is-a-container.md (Topic: docker)
   [2] docker-overview.md (Topic: docker)

[RAG] You: /category fastapi
--> Scoped RAG queries to category: 'fastapi'.

[RAG [Category: fastapi]] You: What is dependency injection?

Assistant: In FastAPI, dependency injection is a design pattern where an object receives
other objects that it depends on. It allows you to declare dependencies in path operation
functions using `Depends()`, making code modular, testable, and reusable.

📚 Sources Cited:
   [1] dependencies.md (Topic: fastapi)

[RAG [Category: fastapi]] You: /general
--> Switched to General mode (general LLM queries).

[GENERAL] You: Tell me a short joke about programming.

Assistant: Why do programmers prefer dark mode? Because light attracts bugs!

[GENERAL] You: /exit
Goodbye! 👋
```

---

### Option C: Run Automated Unit Tests

Run the full unit test suite (25 tests) verifying all 6 modules:

```bash
# Run all unit tests across the entire project
python -m unittest discover tests

# Or run generation tests specifically
python -m unittest tests/test_generation.py
```

#### Test Verifications:
- `test_loaders.py`: Confirms all files are loaded into `Document` objects with non-empty `page_content` and valid `metadata`.
- `test_splitters.py`: Confirms `RecursiveCharacterTextSplitter` generates chunks as `Document` objects, respects max size limits, preserves metadata, and creates valid chunk overlaps.
- `test_embeddings.py`: Confirms LangChain `Embeddings` models produce valid numerical vectors for queries and documents across Google Gemini, OpenAI, and `FakeEmbeddings`.
- `test_vectorstore.py`: Confirms Chroma indexes documents, performs similarity searches, calculates distance scores, supports metadata filtering, and persists to disk.
- `test_retriever.py`: Confirms `BaseRetriever` and `VectorStoreRetriever` implementations for standard similarity, MMR diversity search, threshold filtering, and scoped category retrieval.
- `test_generation.py`: Confirms document formatting (`format_docs`), LCEL RAG chain execution, General conversational chain, and `RAGChatAssistant` multi-mode queries.

#### Expected Output:
```text
.........................
----------------------------------------------------------------------
Ran 25 tests in 2.772s

OK
```

---

## 🧩 Programmatic Usage in Python

You can easily instantiate and use the assistant in your own Python scripts:

```python
from src.generation import create_assistant

# 1. Create the assistant connected to your Chroma database
assistant = create_assistant(persist_directory="./chroma_db")

# 2. RAG Query (Grounded with Sources)
response = assistant.ask(
    query="What is a container image?",
    use_rag=True
)
print("Answer:", response["answer"])
for source in response["sources"]:
    print(f"  Source: {source['file_name']} (Topic: {source['category']})")

# 3. Category-Scoped RAG Query
fastapi_res = assistant.ask(
    query="How does Depends() work in FastAPI?",
    use_rag=True,
    category="fastapi"
)
print("Scoped Answer:", fastapi_res["answer"])

# 4. General Query (Direct LLM without retrieval)
gen_res = assistant.ask(
    query="Explain recursion in simple terms.",
    use_rag=False
)
print("General Answer:", gen_res["answer"])
```
