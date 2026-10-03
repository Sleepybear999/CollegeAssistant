from flask import Flask, request, jsonify, send_from_directory
from rag import answer, retrieve

app = Flask(__name__, static_folder=None)

@app.get("/")
def home():
    return send_from_directory(".", "index.html")

@app.get("/health")
def health():
    return "ok"

@app.post("/ask")
def ask():
    q = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not q:
        return jsonify(error="question is required"), 400
    try:
        sources = [{"title": h["title"], "url": h["url"], "date": h["date"]} for h in retrieve(q, 5)]
        return jsonify(answer=answer(q), sources=sources)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
