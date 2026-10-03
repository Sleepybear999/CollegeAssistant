# Query side for Render. Needs chunks_final.jsonl + vectors.npy in the repo.
import json, os, numpy as np
from fastembed import TextEmbedding
import google.generativeai as genai

rows = [json.loads(l) for l in open("chunks_final.jsonl", encoding="utf-8") if l.strip()]
vecs = np.load("vectors.npy")
assert len(rows) == len(vecs), "chunks and vectors are out of sync"

embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
llm = genai.GenerativeModel("gemini-3.8-flash")  # check current model name

def retrieve(q, k=5):
    qv = next(iter(embedder.query_embed(q)))
    qv = qv / np.linalg.norm(qv)
    idx = np.argsort(vecs @ qv)[-k:][::-1]
    return [rows[i] for i in idx]

def answer(q, k=5):
    hits = retrieve(q, k)
    context = "\n\n".join(f"[{h['title']} | {h['date']}]\n{h['text']}" for h in hits)
    prompt = ("Answer using only this context. If the answer is not in it, say so.\n\n"
              f"Context:\n{context}\n\nQuestion: {q}")
    return llm.generate_content(prompt).text
