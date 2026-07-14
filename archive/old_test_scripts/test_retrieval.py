import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Step 1: Load documents (same as Phase 2)
docs_folder = "policy_docs"
raw_documents = []
for filename in sorted(os.listdir(docs_folder)):
    filepath = os.path.join(docs_folder, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        raw_documents.append({"source": filename, "text": f.read()})

# Step 2: Chunk documents (same settings as Phase 2)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
all_chunks, all_sources, all_ids = [], [], []
chunk_id = 0
for doc in raw_documents:
    for chunk in splitter.split_text(doc["text"]):
        all_chunks.append(chunk)
        all_sources.append(doc["source"])
        all_ids.append(f"chunk_{chunk_id}")
        chunk_id += 1

print(f"Prepared {len(all_chunks)} chunks for storage\n")

# Step 3: Set up ChromaDB with a PERSISTENT client (saves to disk in ./chroma_db)
client = chromadb.PersistentClient(path="./chroma_db")

# Use the SAME embedding model as Phase 1, so everything is consistent
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Delete collection if it exists (so re-running this script doesn't duplicate data)
try:
    client.delete_collection("claims_policies")
except Exception:
    pass

collection = client.create_collection(
    name="claims_policies",
    embedding_function=embedding_fn
)

# Step 4: Add all chunks to the vector database (this embeds them automatically)
collection.add(
    documents=all_chunks,
    metadatas=[{"source": s} for s in all_sources],
    ids=all_ids
)

print("Chunks stored in ChromaDB (saved to ./chroma_db)\n")

# Step 5: Test retrieval with real questions
test_questions = [
    "How long do I have to appeal a denied claim?",
    "What is the deadline for submitting a claim after treatment?",
    "Do planned readmissions count toward the 30-day window?",
    "How many days before a procedure should prior authorization be requested?",
]

for question in test_questions:
    print(f"QUESTION: {question}")
    results = collection.query(query_texts=[question], n_results=2)
    for i in range(len(results["documents"][0])):
        source = results["metadatas"][0][i]["source"]
        distance = results["distances"][0][i]
        text = results["documents"][0][i]
        print(f"  Match {i+1} (from {source}, distance={distance:.3f}):")
        print(f"    {text}")
    print()