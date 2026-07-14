import streamlit as st
import requests

API_URL = "http://127.0.0.1:5000/ask"

# ==========================
# Session State
# ==========================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(
    page_title="Healthcare Claims Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# Sidebar
# ==========================

with st.sidebar:

    st.title("🏥 Claims Assistant")

    st.success("System Status: Online")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.subheader("Technologies")

    st.markdown("""
    - Google Gemini
    - ChromaDB
    - Sentence Transformers
    - Flask API
    - Streamlit
    - Python
    """)

    st.divider()

    st.subheader("Example Questions")

    st.markdown("""
        • How long do I have to appeal a denied claim?

        • Do planned readmissions count?

        • Is prior authorization required?

        • What services require pre-approval?
        """)

    st.divider()

    st.caption(
        "Healthcare Claims RAG Assistant\n\nVersion 1.0"
    )

st.title("🏥 Healthcare Claims Policy Assistant")

st.markdown(
"""
AI-powered Retrieval-Augmented Generation (RAG)
assistant for healthcare claims policy documents.

Ask questions about policy coverage, appeals,
readmissions, prior authorization,
and related healthcare claims.
"""
)

st.divider()


question = st.text_area(
    "Enter your healthcare policy question",
    placeholder="Example: How long do I have to appeal a denied claim?",
    height=120
)

if st.button("Get Answer"):

    if not question.strip():
        st.warning("Please enter a question.")
    else:

        with st.spinner("🔍 Searching policy documents..."):

            try:

                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=60
                )

                result = response.json()

                if response.status_code == 200:

                    st.session_state.chat_history.append({

                        "question": question,

                        "answer": result["answer"],

                        "confidence": result["confidence"],

                        "sources": result["sources"],

                        "chunks": result["chunks"],

                        "distances": result["distances"],

                        "best_distance": result["best_distance"],

                        "chunks_used": result["chunks_used"],

                        "documents_used": result["documents_used"]

                    })

                    st.success("Answer generated successfully using Retrieval-Augmented Generation.")

                    # st.subheader("Answer")
                    # st.write(result["answer"])

                    # st.subheader("Confidence")
                    # st.info(result["confidence"])

                    # st.subheader("Sources")

                    # if result["sources"]:
                    #     for source in result["sources"]:
                    #         st.write(f"• {source}")
                    # else:
                    #     st.write("No sources returned.")

                    # st.subheader("Best Distance")
                    # st.write(round(result["best_distance"], 3))

                else:
                    st.error(result.get("error", "Unknown API error"))

            except Exception as e:
                st.error(f"Could not connect to API.\n\n{e}")
                # ==========================
                # Conversation History
                # ==========================

if st.session_state.chat_history:

    st.divider()

    st.header("💬 Conversation")

    for chat in reversed(st.session_state.chat_history):

        with st.chat_message("user"):

            st.markdown(chat["question"])

        with st.chat_message("assistant"):

            st.markdown(chat["answer"])
            if chat["confidence"] == "HIGH":
                st.success("Confidence : HIGH")

            elif chat["confidence"] == "MEDIUM":
                st.warning("Confidence : MEDIUM")

            else:
                st.error("Confidence : LOW")

            st.caption(
                f"Best Distance: {chat['best_distance']:.3f}"
            )

            st.caption(
                f"Chunks Retrieved: {chat['chunks_used']}"
            )

            st.caption(
                f"Documents Used: {chat['documents_used']}"
            )


        col1, col2 = st.columns(2)

        with col1:

            if chat["confidence"] == "HIGH":
                st.success("🟢 HIGH")

            elif chat["confidence"] == "MEDIUM":
                st.warning("🟡 MEDIUM")

            else:
                st.error("🔴 LOW")

        with col2:

            similarity = max(0, (1 - chat["best_distance"]) * 100)

            st.info(
                f"Similarity: {similarity:.1f}%"
            )

        if chat["sources"]:

            smetric1, metric2 = st.columns(2)

            with metric1:
                st.metric(
                    "Retrieved Documents",
                    chat["documents_used"]
                )

            with metric2:
                st.metric(
                    "Retrieved Chunks",
                    chat["chunks_used"]
                )

        with st.expander("🔍 Retrieval Evaluation"):

            for i in range(len(chat["chunks"])):

                st.markdown(f"### Rank #{i+1}")

                if i < len(chat["sources"]):

                    st.write(
                        f"**Source:** {chat['sources'][i]}"
                    )

                distance = chat["distances"][i]

                if distance < 0.20:

                    st.success(
                        f"Excellent Match ({distance:.3f})"
                    )

                elif distance < 0.35:

                    st.info(
                        f"Good Match ({distance:.3f})"
                    )

                else:

                    st.warning(
                        f"Weak Match ({distance:.3f})"
                    )

                    st.code(
                        chat["chunks"][i],
                        language="text"
                    )

                st.divider()
else:

    st.divider()

    st.info(
        "💬 No conversations yet.\n\nAsk your first healthcare policy question."
    )                