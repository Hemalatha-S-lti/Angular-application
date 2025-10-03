from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os

app = Flask(__name__)
CORS(app)

# -------------------------
# Set your Azure OpenAI info
# -------------------------
openai.api_type = "azure"
openai.api_base = "https://YOUR_AZURE_OPENAI_ENDPOINT/"
openai.api_version = "2024-05-01"  # use your deployment version
openai.api_key = "YOUR_API_KEY"

deployment_name = "YOUR_DEPLOYMENT_NAME"  # the deployment name of your model

# -------------------------
# API route for frontend
# -------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    
    if not question:
        return jsonify({"answer": "No question provided."})
    
    try:
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=[{"role": "user", "content": question}],
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})
    
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})

# -------------------------
# Run backend
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
