from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import numpy as np

app = Flask(__name__)
CORS(app)

# -------------------------
# Azure OpenAI Configuration
# -------------------------
openai.api_type = "azure"
openai.api_base = "https://YOUR_AZURE_OPENAI_ENDPOINT/"
openai.api_version = "2024-05-01"
openai.api_key = "YOUR_API_KEY"

deployment_name = "YOUR_DEPLOYMENT_NAME"  # GPT deployment
embedding_model = "text-embedding-3-small"  # embedding model

# -------------------------
# Example documents
# -------------------------
documents = [
    {"id": 1, "text": "Angular is a frontend framework by Google."},
    {"id": 2, "text": "Flask is a lightweight Python web framework."},
    {"id": 3, "text": "Azure OpenAI service allows GPT models in the cloud."}
]

# -------------------------
# Precompute embeddings for documents
# -------------------------
document_embeddings = []
for doc in documents:
    emb = openai.Embedding.create(
        input=doc["text"],
        model=embedding_model
    )['data'][0]['embedding']
    document_embeddings.append({"id": doc["id"], "text": doc["text"], "embedding": emb})

# -------------------------
# Cosine similarity function
# -------------------------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -------------------------
# API route for frontend
# -------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"answer": "No question provided."})
    
    # Step 1: Get embedding of the question
    q_emb = openai.Embedding.create(
        input=question,
        model=embedding_model
    )['data'][0]['embedding']
    
    # Step 2: Find most similar document
    similarities = [cosine_similarity(q_emb, doc["embedding"]) for doc in document_embeddings]
    best_index = np.argmax(similarities)
    top_doc = document_embeddings[best_index]["text"]
    
    # Step 3: Ask GPT with the top document as context
    try:
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Answer the question based on this document: {top_doc}\nQuestion: {question}"}
            ],
            temperature=0.7
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})
    
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
