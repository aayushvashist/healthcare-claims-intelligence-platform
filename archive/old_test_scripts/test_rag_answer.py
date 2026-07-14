from dotenv import load_dotenv
import os
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

llm = genai.GenerativeModel("gemini-2.0-flash")

# Reconnect to your existing ChromaDB (built in Phase 3, saved on disk)
client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_collection(name="claims_policies", embedding_function=embedding_fn)

# Confidence threshold - if the BEST match's distance is above this, we don't trust it
CONFIDENCE_THRESHOLD = 0.40

def answer_question(question, n_results=2):
    # Step 1: Retrieve relevant chunks (same as Phase 3)
    results = collection.query(query_texts=[question], n_results=n_results)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    distances = results["distances"][0]

    best_distance = distances[0]

    # Step 2: Confidence check BEFORE calling the LLM
    if best_distance > CONFIDENCE_THRESHOLD:
        return {
            "answer": "I don't have enough confident information in the policy documents to answer this accurately. Please consult the full policy or a specialist.",
            "sources": [],
            "confidence": "LOW",
            "best_distance": best_distance
        }

    # Step 3: Build the context block from retrieved chunks
    context = "\n\n".join([f"[Source: {src}]\n{chunk}" for src, chunk in zip(sources, chunks)])

    # Step 4: Build the prompt - this is the most important part of RAG
    prompt = f"""You are a healthcare claims policy assistant. Answer the question using ONLY the information in the context below. Do not use any outside knowledge. If the context does not contain the answer, say so clearly.

Context:
{context}

Question: {question}

Answer concisely, and mention which policy the answer comes from."""

    # Step 5: Call the LLM
    response = llm.generate_content(prompt)

    return {
        "answer": response.text,
        "sources": list(set(sources)),
        "confidence": "HIGH" if best_distance < 0.25 else "MEDIUM",
        "best_distance": best_distance
    }

# ---- TEST IT ----
test_questions = [
    "How long do I have to appeal a denied claim?",
    "Do planned readmissions count toward the 30-day window?",
    "What is the capital of France?",  # deliberately irrelevant - tests confidence flag
]

for q in test_questions:
    result = answer_question(q)
    print(f"QUESTION: {q}")
    print(f"CONFIDENCE: {result['confidence']} (distance={result['best_distance']:.3f})")
    print(f"SOURCES: {result['sources']}")
    print(f"ANSWER: {result['answer']}")
    print("-" * 80)