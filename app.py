from flask import Flask, request, jsonify, send_from_directory
from rag import answer, rewrite
from pyq import search_papers

app = Flask(__name__, static_folder=None)

@app.get("/")
def home():
    return send_from_directory(".", "index.html")

@app.get("/health")
def health():
    return "ok"

@app.post("/ask")
def ask():
    body = request.get_json(silent=True) or {}
    q = body.get("question", "").strip()
    history = [m for m in body.get("history", []) if isinstance(m, dict) and m.get("role") and m.get("text")][-6:]
    if not q:
        return jsonify(error="question is required"), 400
    try:
        standalone = rewrite(history, q)          # resolves "what about end sem?" etc.
        papers = search_papers(standalone)
        if papers is None and standalone != q:
            papers = search_papers(q)
        if papers is not None:
            return jsonify(answer=papers["message"], groups=papers["groups"])
        text, hits = answer(standalone, history)
        sources = [{"title": h["title"], "url": h["url"], "date": h["date"]} for h in hits]
        return jsonify(answer=text, sources=sources)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
