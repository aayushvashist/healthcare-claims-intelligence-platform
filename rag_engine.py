"""
Healthcare Claims RAG Engine

Responsibilities
----------------
1. Retrieve relevant policy chunks from ChromaDB
2. Build an LLM prompt
3. Generate answer using Gemini
4. Return structured response for Flask API
"""

import os
import time
import logging

from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions

from google import genai

# ============================================================
# Logging Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found inside .env")

# ============================================================
# Project Constants
# ============================================================

MODEL_NAME = "gemini-2.0-flash"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

COLLECTION_NAME = "healthcare_claims"

CHROMA_DB_PATH = "./chroma_db"

CONFIDENCE_THRESHOLD = 0.40

TOP_K_RESULTS = 3

MAX_RETRIES = 3

RETRY_DELAY = 2

# ============================================================
# Initialize Gemini
# ============================================================

logger.info("Initializing Gemini Client...")

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

# ============================================================
# Initialize ChromaDB
# ============================================================

logger.info("Connecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH
)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

logger.info("ChromaDB Loaded Successfully")

# ============================================================
# Retrieve Documents
# ============================================================

def retrieve_documents(question, n_results=TOP_K_RESULTS):

    logger.info("Searching Vector Database...")

    results = collection.query(

        query_texts=[question],

        n_results=n_results

    )

    logger.info("Retrieved %d chunks", len(results["documents"][0]))

    return results


# ============================================================
# Prompt Builder
# ============================================================

def build_prompt(question, chunks, sources):

    context = "\n\n".join(

        f"[Source: {source}]\n{chunk}"

        for source, chunk in zip(sources, chunks)

    )

    prompt = f"""
You are an expert Healthcare Claims Policy Assistant.

You MUST answer ONLY using the policy context below.

Rules:

- Never use outside knowledge.
- Never fabricate information.
- Never guess.
- If information is unavailable in the policy,
  clearly say so.
- Answer professionally.
- Prefer bullet points whenever appropriate.
- Mention the policy document(s) used.

========================================
POLICY CONTEXT
========================================

{context}

========================================
QUESTION
========================================

{question}

========================================
ANSWER
========================================
"""

    return prompt


# ============================================================
# Gemini Generation
# ============================================================

def generate_answer(prompt):

    for attempt in range(MAX_RETRIES):

        try:

            logger.info(
                "Calling Gemini (Attempt %d)...",
                attempt + 1
            )

            response = gemini_client.models.generate_content(

                model=MODEL_NAME,

                contents=prompt

            )

            logger.info("Gemini Response Generated")

            return response.text

        except Exception as e:

            error_message = str(e)

            logger.warning(error_message)

            if attempt < MAX_RETRIES - 1:

                logger.info(
                    "Retrying in %d seconds...",
                    RETRY_DELAY
                )

                time.sleep(RETRY_DELAY)

            else:

                raise RuntimeError(
                    "Gemini service is temporarily unavailable. Please try again later."
                )


# ============================================================
# Complete RAG Pipeline
# ============================================================

def answer_question(question):

    results = retrieve_documents(question)

    chunks = results["documents"][0]

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]

    sources = [metadata["source"] for metadata in metadatas]

    best_distance = distances[0]

    logger.info(
        "Best Similarity Distance : %.3f",
        best_distance
    )

    # --------------------------------------------------------

    if best_distance > CONFIDENCE_THRESHOLD:

        logger.warning("Low Confidence Retrieval")

        return {

            "answer":
                "The retrieved policy documents do not contain enough reliable information to answer this question. Please consult the complete CMS policy document or a healthcare claims specialist.",

            "sources": [],

            "chunks": [],

            "distances": [],  

            "confidence": "LOW",

            "best_distance": best_distance,

            "chunks_used": len(chunks),

            "documents_used": 0

        }

    # --------------------------------------------------------

    prompt = build_prompt(

        question,

        chunks,

        sources

    )

    answer = generate_answer(prompt)

    # --------------------------------------------------------

    if best_distance < 0.20:

        confidence = "HIGH"

    elif best_distance < 0.35:

        confidence = "MEDIUM"

    else:

        confidence = "LOW"

    unique_sources = list(dict.fromkeys(sources))

    logger.info(
        "Answer generated successfully using %d chunks from %d document(s).",
        len(chunks),
        len(unique_sources)
    )

    return {

        "answer": answer,

        "sources": unique_sources,

        "chunks": chunks,

        "distances": distances,

        "confidence": confidence,

        "best_distance": best_distance,

        "chunks_used": len(chunks),

        "documents_used": len(unique_sources)

    }


# ============================================================
# Standalone Testing
# ============================================================

if __name__ == "__main__":

    question = "How long do I have to appeal a denied claim?"

    result = answer_question(question)

    print("\nQuestion")

    print(question)

    print("\nAnswer")

    print(result["answer"])

    print("\nConfidence")

    print(result["confidence"])

    print("\nSources")

    print(result["sources"])