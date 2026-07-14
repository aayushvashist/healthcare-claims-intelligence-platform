
# 🏥 Healthcare Claims Intelligence Platform

> **AI-Powered Retrieval-Augmented Generation (RAG) System for Healthcare Policy Documents**

An end-to-end Retrieval-Augmented Generation (RAG) application that indexes official CMS healthcare policy documents and answers healthcare policy questions using **Google Gemini**, **ChromaDB**, **Sentence Transformers**, **Flask REST API**, and **Streamlit**.

---

## 🚀 Project Overview

Healthcare organizations rely on lengthy policy documents that are difficult to search efficiently using traditional keyword search.

This project demonstrates how modern AI techniques can combine:

- Semantic Search
- Vector Databases
- Large Language Models
- Prompt Engineering

to build an intelligent Healthcare Claims Assistant capable of retrieving relevant policy information before generating grounded answers.

The knowledge base is built from CMS healthcare policy documents, indexed into ChromaDB using Sentence Transformer embeddings.

---

# ✨ Features

- ✅ Retrieval-Augmented Generation (RAG)
- ✅ Official CMS policy knowledge base
- ✅ Semantic Search using Sentence Transformers
- ✅ ChromaDB vector database
- ✅ Google Gemini integration
- ✅ Flask REST API
- ✅ Streamlit frontend
- ✅ Conversation history
- ✅ Confidence scoring
- ✅ Retrieval evaluation dashboard
- ✅ Source attribution
- ✅ Modular architecture
- ✅ Automatic vector database builder

---

# 🏗️ System Architecture

```text
                User
                  │
                  ▼
        Streamlit Frontend
                  │
                  ▼
           Flask REST API
                  │
                  ▼
             rag_engine.py
          ┌────────┴────────┐
          ▼                 ▼
     Google Gemini      ChromaDB
                              ▲
                              │
                  build_vector_db.py
                              ▲
                              │
                 CMS Policy Documents
```

---

# 🔄 Workflow

1. User asks a healthcare policy question.
2. ChromaDB retrieves the most relevant document chunks.
3. Prompt is built using retrieved context.
4. Gemini generates an answer grounded in retrieved evidence.
5. Streamlit displays the answer, confidence, metrics, and retrieved sources.

---

# 🧰 Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.11 |
| LLM | Google Gemini 2.0 Flash |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Backend | Flask |
| Frontend | Streamlit |
| API Testing | Postman |
| Environment | python-dotenv |

---

# 📁 Project Structure

```text
claims-qa-assistant/
│
├── api.py
├── rag_engine.py
├── build_vector_db.py
├── streamlit_app.py
├── chat.py
├── policy_docs/
├── chroma_db/
├── archive/
├── README.md
├── requirements.txt
└── .gitignore
```

---

# ⚙️ Installation

```bash
git clone <repository-url>

cd claims-qa-assistant

python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

---

# 🧠 Build the Vector Database

```bash
python build_vector_db.py
```

This script:

- Reads CMS policy documents
- Splits them into chunks
- Creates embeddings
- Stores vectors in ChromaDB

---

# ▶️ Run the Application

Start the Flask API

```bash
python api.py
```

Open another terminal

```bash
streamlit run streamlit_app.py
```

---

# 🌐 REST API

## POST /ask

Request

```json
{
  "question":"How long do I have to appeal a denied claim?"
}
```

Response

```json
{
  "answer":"...",
  "confidence":"HIGH",
  "best_distance":0.18,
  "sources":["cms_appeals_first_level.txt"]
}
```

---

# 📊 Retrieval Evaluation Dashboard

The Streamlit interface displays:

- Confidence Level
- Similarity Score
- Retrieved Documents
- Retrieved Chunks
- Retrieved Source Files
- Chunk Viewer

This improves explainability and debugging of the RAG pipeline.

---

# 📸 Screenshots

Add screenshots after deployment:

```text
assets/
│
├── home.png
├── conversation.png
├── retrieval_dashboard.png
└── architecture.png
```

---

# 🚀 Future Enhancements

- PDF Upload
- Automatic PDF Ingestion
- OCR Support
- Multiple Knowledge Bases
- User Authentication
- Docker
- Cloud Deployment
- Retrieval Analytics
- Multi-turn Context Memory

---

# 💼 Resume Highlights

This project demonstrates experience with:

- Retrieval-Augmented Generation (RAG)
- Vector Databases
- Semantic Search
- Prompt Engineering
- REST APIs
- Streamlit
- LLM Integration
- AI Application Development
- Healthcare Policy Analytics

---

# 👨‍💻 Author

**Ayush Vashist**

MBA (Business Analytics)

Business Development Manager – SBI General Insurance

Aspiring Data Analyst | AI Engineer

---

# 📄 License

This project is intended for educational and portfolio purposes. CMS documents remain the property of their respective publishers.
