import re
from flask import Flask, request, jsonify, send_from_directory
from rag import answer, rewrite
from pyq import search_papers

app = Flask(__name__, static_folder=None)

SRC_MIN = 0.55   # only show a source link if its similarity score is at least this (tune 0.5-0.65)
SMALLTALK = re.compile(r"^\s*(hi+|hey+|hello+|yo|good\s+(morning|afternoon|evening)|thanks?|thank\s+you|ok(ay)?|bye)\s*[!.?]*\s*$", re.I)

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
    if SMALLTALK.match(q):
        return jsonify(answer="Hi! Ask me anything about NIT Jalandhar, or search papers like \u201cDBMS mid sem paper\u201d.")
    try:
        # 1) Paper search on the user's exact words first (fast, no Gemini call)
        papers = search_papers(q)
        if papers and papers["groups"]:
            return jsonify(answer=papers["message"], groups=papers["groups"])
        # 2) Follow-up? Rewrite with chat history and try again
        standalone = rewrite(history, q) if history else q
        if standalone != q:
            p2 = search_papers(standalone)
            if p2 and p2["groups"]:
                return jsonify(answer=p2["message"], groups=p2["groups"])
        if papers is not None:               # paper request, but nothing found
            return jsonify(answer=papers["message"], groups=[])
        # 3) Normal question -> RAG
        text, hits = answer(standalone, history)
        sources = [{"title": h["title"], "url": h["url"], "date": h["date"]} for h in hits if h["score"] >= SRC_MIN][:3]
        return jsonify(answer=text, sources=sources)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
