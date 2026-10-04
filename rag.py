# Query side for Render. Needs chunks_final.jsonl + vectors.npy in the repo.
import json, os, numpy as np
from fastembed import TextEmbedding
from google import genai

def _fix(s):
    """Repair UTF-8 text that was mis-decoded as latin-1 (e.g. 'à¤¸' -> Devanagari)."""
    if isinstance(s, str) and ("Ã" in s or "à" in s):
        try:
            return s.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return s

rows = []
for l in open("chunks_final.jsonl", encoding="utf-8"):
    if l.strip():
        d = json.loads(l)
        d["title"], d["text"] = _fix(d.get("title", "")), _fix(d.get("text", ""))
        rows.append(d)
vecs = np.load("vectors.npy")
assert len(rows) == len(vecs), "chunks and vectors are out of sync"

embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.5-flash-lite"

def retrieve(q, k=5):
    qv = next(iter(embedder.query_embed(q)))
    qv = qv / np.linalg.norm(qv)
    sims = vecs @ qv
    idx = np.argsort(sims)[-k:][::-1]
    return [{**rows[i], "score": float(sims[i])} for i in idx]

def rewrite(history, q):
    """Turn a follow-up into a standalone search query using recent chat history."""
    if not history:
        return q
    convo = "\n".join(f"{m['role']}: {m['text'][:300]}" for m in history[-6:])
    prompt = (
        "Rewrite the user's last message as one standalone search query, using the conversation only to fill "
        "in what it refers to. Fix spelling mistakes. Never drop any word the user wrote and never add place or "
        "institute names. Expand abbreviations (e.g. DBMS -> database management system). If they want exam "
        "papers, write it like: '<subject> <mid sem|end sem> paper <year>'. "
        "Output only the rewritten query, nothing else.\n\n"
        f"Conversation:\n{convo}\nuser: {q}\n\nRewritten query:"
    )
    try:
        out = client.models.generate_content(model=MODEL, contents=prompt).text.strip()
        return out.splitlines()[0][:300] or q
    except Exception:
        return q

def answer(q, history=None, k=6):
    hits = retrieve(q, k)
    context = "\n\n".join(f"[{h['title']} | {h['date']}]\n{h['text']}" for h in hits)
    convo = "\n".join(f"{m['role']}: {m['text'][:400]}" for m in (history or [])[-6:])
    prompt = ("You are a helpful assistant for NIT Jalandhar students. Answer using only the context below. "
              "If the answer is not in the context, say you don't have that information. "
              "Use the conversation only to understand what the question refers to.\n\n"
              f"Conversation so far:\n{convo or '(none)'}\n\nContext:\n{context}\n\nQuestion: {q}")
    text = client.models.generate_content(model=MODEL, contents=prompt).text
    return text, hits
