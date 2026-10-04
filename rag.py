# Query side for Render. Needs chunks_final.jsonl + vectors.npy in the repo.
import json, os, numpy as np
from fastembed import TextEmbedding
from google import genai

rows = [json.loads(l) for l in open("chunks_final.jsonl", encoding="utf-8") if l.strip()]
vecs = np.load("vectors.npy")
assert len(rows) == len(vecs), "chunks and vectors are out of sync"

embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.5-flash-lite"  # check current model name

def retrieve(q, k=5):
    qv = next(iter(embedder.query_embed(q)))
    qv = qv / np.linalg.norm(qv)
    idx = np.argsort(vecs @ qv)[-k:][::-1]
    return [rows[i] for i in idx]

def rewrite(history, q):
    """Turn a follow-up into a standalone search query using recent chat history."""
    if not history:
        return q
    convo = "\n".join(f"{m['role']}: {m['text'][:300]}" for m in history[-6:])
    prompt = (
        "Rewrite the user's last message as one standalone search query for a college (NIT Jalandhar) "
        "assistant, using the conversation for context. Expand abbreviations (e.g. DBMS -> database management "
        "system). If they want exam papers, write it like: '<full subject name> <mid sem|end sem> paper <year>'. "
        "Keep every detail they mentioned. Output only the rewritten query, nothing else.\n\n"
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
