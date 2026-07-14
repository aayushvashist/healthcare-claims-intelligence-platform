from sentence_transformers import SentenceTransformer, util

# Load a small, free, pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Three test sentences
sentences = [
    "The patient was readmitted within 30 days",
    "Patient came back to hospital in a month",
    "The weather is sunny today"
]

# Convert each sentence into its number-list (embedding)
embeddings = model.encode(sentences)

# Print how long each embedding is
print("Each sentence becomes a list of", len(embeddings[0]), "numbers")
print("First 5 numbers of sentence 1:", embeddings[0][:5])

# Now measure similarity between sentences
sim_1_2 = util.cos_sim(embeddings[0], embeddings[1])
sim_1_3 = util.cos_sim(embeddings[0], embeddings[2])

print(f"\nSimilarity between sentence 1 & 2 (related meaning): {sim_1_2.item():.3f}")
print(f"Similarity between sentence 1 & 3 (unrelated): {sim_1_3.item():.3f}")