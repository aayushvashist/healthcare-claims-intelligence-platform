from flask import Flask, request, jsonify

from rag_engine import answer_question

app = Flask(__name__)


@app.route("/")
def home():
    return jsonify({
        "message": "Healthcare Claims RAG API",
        "status": "running"
    })


@app.route("/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "JSON body required."
            }), 400

        question = data.get("question")

        if not question:

            return jsonify({
                "error": "Question is required."
            }), 400

        result = answer_question(question)

        return jsonify({

            "answer": result["answer"],

            "sources": result["sources"],

            "chunks": result["chunks"],

            "distances": result["distances"],

            "confidence": result["confidence"],

            "best_distance": result["best_distance"],

            "chunks_used": result["chunks_used"],

            "documents_used": result["documents_used"]

        })
    
    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )