from rag_engine import answer_question

print("=" * 60)
print("Healthcare Claims Policy Assistant")
print("Type 'exit' anytime to quit.")
print("=" * 60)

while True:

    question = input("\nAsk a question: ").strip()

    if question.lower() == "exit":
        print("\nGoodbye!")
        break

    if not question:
        print("Please enter a question.")
        continue

    try:
        result = answer_question(question)

        print("\nAnswer")
        print("-" * 50)
        print(result["answer"])

        print("\nConfidence:", result["confidence"])
        print("Sources:", ", ".join(result["sources"]))

    except Exception as e:
        print("\nError:", e)