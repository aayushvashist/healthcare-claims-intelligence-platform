import os
import shutil

import chromadb

from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

# ==========================================
# Configuration
# ==========================================

DOCS_FOLDER = "policy_docs"

CHROMA_DB_DIR = "chroma_db"

COLLECTION_NAME = "healthcare_claims"

CHUNK_SIZE = 300

CHUNK_OVERLAP = 50

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ==========================================
# Remove old database
# ==========================================

if os.path.exists(CHROMA_DB_DIR):

    print("Deleting old vector database...")

    shutil.rmtree(CHROMA_DB_DIR)

# ==========================================
# Create Chroma Client
# ==========================================

client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

# ==========================================
# Chunk Splitter
# ==========================================

splitter = RecursiveCharacterTextSplitter(

    chunk_size=CHUNK_SIZE,

    chunk_overlap=CHUNK_OVERLAP,

    separators=["\n\n", "\n", ". ", " ", ""]
)

# ==========================================
# Statistics
# ==========================================

documents_indexed = 0

chunks_indexed = 0

# ==========================================
# Process every document
# ==========================================

for filename in sorted(os.listdir(DOCS_FOLDER)):

    if not filename.endswith(".txt"):

        continue

    filepath = os.path.join(DOCS_FOLDER, filename)

    print(f"\nProcessing {filename}")

    with open(filepath, "r", encoding="utf-8") as f:

        text = f.read()

    chunks = splitter.split_text(text)

    print(f"Chunks created: {len(chunks)}")

    documents = []

    metadatas = []

    ids = []

    for i, chunk in enumerate(chunks):

        documents.append(chunk)

        metadatas.append({

            "source": filename,

            "chunk": i

        })

        ids.append(f"{filename}_{i}")

    collection.add(

        documents=documents,

        metadatas=metadatas,

        ids=ids

    )

    documents_indexed += 1

    chunks_indexed += len(chunks)

# ==========================================
# Finished
# ==========================================

print("\n===================================")

print("Vector Database Created Successfully")

print("===================================")

print(f"Documents Indexed : {documents_indexed}")

print(f"Chunks Indexed    : {chunks_indexed}")

print(f"Embedding Model   : {EMBEDDING_MODEL}")

print(f"Database Folder   : {CHROMA_DB_DIR}")