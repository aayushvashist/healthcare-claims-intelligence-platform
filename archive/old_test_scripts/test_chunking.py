from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Step 1: Load all the policy documents
docs_folder = "policy_docs"
raw_documents = []

for filename in sorted(os.listdir(docs_folder)):
    filepath = os.path.join(docs_folder, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
        raw_documents.append({"source": filename, "text": text})

print(f"Loaded {len(raw_documents)} documents\n")

# Step 2: Set up the chunker
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Step 3: Chunk every document, tracking source
all_chunks = []
for doc in raw_documents:
    chunks = splitter.split_text(doc["text"])
    for chunk in chunks:
        all_chunks.append({"source": doc["source"], "text": chunk})

print(f"Split into {len(all_chunks)} total chunks\n")

# Step 4: Print every chunk so you can inspect all of them
for i, c in enumerate(all_chunks):
    print(f"--- Chunk {i+1} (from {c['source']}, {len(c['text'])} chars) ---")
    print(c["text"])
    print()