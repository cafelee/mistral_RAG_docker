# llm_container/llm_api.py
from flask import Flask, request, jsonify
from util.llm_rag import build_qa_pipeline

app = Flask(__name__)
qa_chain = build_qa_pipeline()

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    answer = qa_chain.run(query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
