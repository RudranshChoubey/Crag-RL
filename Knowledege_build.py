"""Step 4 - Build the English Wikipedia knowledge base (ChromaDB).

Changes from the first version:
  * 120-word chunks (20 overlap) instead of 350: the MiniLM encoder reads only 128 word-pieces,
    so most of a 350-word chunk was never embedded.
  * ALL of Simple English Wikipedia by default (was the first 25,000 articles).
  * XQuAD English paragraphs are indexed, so the training/validation questions are answerable.
  * --targeted also fetches full English Wikipedia pages for every TruthfulQA question and caches
    them in data/kb_cache/ (a frozen, reproducible snapshot).
  * New collection name (wiki_passages_v2) so old 350-word chunks are not mixed in.

    python Knowledege_build.py --targeted                 # recommended (~30-60 min on GPU)
    python Knowledege_build.py --max-articles 50000       # faster, less coverage
"""
import argparse
import json
import os
import time

from crag_utils import (CHROMA_DIR, COLLECTION, ENCODER_NAME, KB_CACHE_DIR, TQA_CSV, TQA_CSV_URL, get_device)

BATCH = 256


def chunk_text(text, chunk_words=120, overlap=20, min_words=25):
    words = text.split()
    step = max(1, chunk_words - overlap)
    chunks = []
    for i in range(0, len(words), step):
        piece = words[i:i + chunk_words]
        if len(piece) >= min_words or (i == 0 and piece):
            chunks.append(" ".join(piece))
        if i + chunk_words >= len(words):
            break
    return chunks


def simple_wiki_docs(max_articles):
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)
    for i, art in enumerate(ds):
        if max_articles and i >= max_articles:
            break
        if art.get("text", "").strip():
            yield {"id": f"simple_{art['id']}", "title": art["title"], "text": art["text"], "source": "simplewiki"}


def xquad_docs():
    from datasets import load_dataset

    ds = load_dataset("google/xquad", "xquad.en", split="validation")
    seen = set()
    for row in ds:
        ctx = row["context"]
        if ctx in seen:
            continue
        seen.add(ctx)
        yield {"id": f"xquad_{len(seen)}", "title": "", "text": ctx, "source": "xquad"}


def _get_json(session, url, params, retries=5):
    """GET with backoff. Wikipedia answers blocked or rate-limited requests with an HTML page,
    which is what caused the JSONDecodeError; this retries and reports the HTTP status instead."""
    for attempt in range(retries):
        r = session.get(url, params=params, timeout=30)
        if r.status_code == 200 and "json" in r.headers.get("Content-Type", ""):
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            wait = int(r.headers.get("Retry-After", 0) or 0) or 2 ** (attempt + 1)
            time.sleep(min(wait, 60))
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200].strip()}")
    raise RuntimeError(f"HTTP {r.status_code} after {retries} retries")


def targeted_docs(contact, per_query=2, max_words=2500, delay=0.5):
    """Full English Wikipedia pages for each TruthfulQA question (cached = frozen snapshot)."""
    import pandas as pd
    import requests

    os.makedirs(KB_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(KB_CACHE_DIR, "targeted_pages.jsonl")
    done_path = os.path.join(KB_CACHE_DIR, "targeted_queries_done.json")
    pages = {}
    if os.path.exists(cache_path):
        for l in open(cache_path, encoding="utf-8"):
            d = json.loads(l)
            pages[d["pageid"]] = d
    done = set(json.load(open(done_path, encoding="utf-8"))) if os.path.exists(done_path) else set()
    src = TQA_CSV if os.path.exists(TQA_CSV) else TQA_CSV_URL
    questions = list(pd.read_csv(src)["Question"].astype(str))
    api = "https://en.wikipedia.org/w/api.php"
    s = requests.Session()
    # Wikimedia's User-Agent policy requires contact information (a URL or e-mail) for automated clients.
    s.headers.update({"User-Agent": f"CRAG-RL-research/1.0 ({contact}) python-requests"})
    print(f"  {len(done)}/{len(questions)} questions already fetched, {len(pages)} pages cached")
    failures = 0
    with open(cache_path, "a", encoding="utf-8") as fout:
        for i, q in enumerate(questions):
            if q in done:
                continue
            try:
                res = _get_json(s, api, {"action": "query", "list": "search", "srsearch": q,
                                         "srlimit": per_query, "format": "json"})
                for h in res.get("query", {}).get("search", []):
                    pid = h["pageid"]
                    if pid in pages:
                        continue
                    e = _get_json(s, api, {"action": "query", "prop": "extracts", "explaintext": 1,
                                           "pageids": pid, "format": "json"})
                    page = e["query"]["pages"][str(pid)]
                    d = {"pageid": pid, "title": page.get("title", ""),
                         "text": " ".join(page.get("extract", "").split()[:max_words])}
                    pages[pid] = d
                    fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                    fout.flush()
                    time.sleep(delay)
                done.add(q)
                failures = 0
            except Exception as ex:
                failures += 1
                print(f"  [!] skipped ({ex}): {q[:60]}")
                if failures >= 5:
                    print("  [!] 5 failures in a row - stopping the fetch. Rerun later; finished questions are kept.")
                    break
            time.sleep(delay)
            if (i + 1) % 25 == 0:
                print(f"  targeted: {i + 1}/{len(questions)} questions, {len(pages)} pages")
                json.dump(sorted(done), open(done_path, "w", encoding="utf-8"))
    json.dump(sorted(done), open(done_path, "w", encoding="utf-8"))
    print(f"  {len(done)}/{len(questions)} questions fetched, {len(pages)} pages cached in {cache_path}")
    for d in pages.values():
        if d["text"]:
            yield {"id": f"enwiki_{d['pageid']}", "title": d["title"], "text": d["text"], "source": "enwiki_targeted"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-articles", type=int, default=0, help="0 = all of Simple English Wikipedia")
    ap.add_argument("--chunk-words", type=int, default=120)
    ap.add_argument("--overlap", type=int, default=20)
    ap.add_argument("--no-simplewiki", action="store_true")
    ap.add_argument("--no-xquad", action="store_true")
    ap.add_argument("--targeted", action="store_true", help="add full enwiki pages for TruthfulQA questions")
    ap.add_argument("--contact", default="https://github.com",
                    help="URL or e-mail for the Wikipedia User-Agent (Wikimedia policy), e.g. your GitHub repo URL")
    ap.add_argument("--reset", action="store_true", help="delete the collection first")
    a = ap.parse_args()

    import chromadb
    from sentence_transformers import SentenceTransformer

    device = get_device()
    print(f"[*] Encoder {ENCODER_NAME} on {device}")
    enc = SentenceTransformer(ENCODER_NAME, device=device)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if a.reset:
        try:
            client.delete_collection(COLLECTION)
            print(f"[*] Deleted collection {COLLECTION}")
        except Exception:
            pass
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    ids, docs, metas = [], [], []
    total = 0

    def flush():
        nonlocal ids, docs, metas, total
        if not ids:
            return
        embs = enc.encode(docs, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        col.upsert(ids=ids, documents=docs, embeddings=embs.tolist(), metadatas=metas)
        total += len(ids)
        print(f"[+] Indexed {total} passages ...", end="\r")
        ids, docs, metas = [], [], []

    sources = []
    if not a.no_xquad:
        sources.append(("XQuAD English paragraphs", xquad_docs()))
    if a.targeted:
        sources.append(("targeted English Wikipedia pages", targeted_docs(a.contact)))
    if not a.no_simplewiki:
        sources.append(("Simple English Wikipedia", simple_wiki_docs(a.max_articles)))

    for name, gen in sources:
        print(f"\n[*] Indexing {name} ...")
        for d in gen:
            for j, ch in enumerate(chunk_text(d["text"], a.chunk_words, a.overlap)):
                ids.append(f"{d['id']}_{j}")
                docs.append(f"Title: {d['title']}. {ch}" if d["title"] else ch)
                metas.append({"title": d["title"], "source": d["source"]})
                if len(ids) >= BATCH:
                    flush()
        flush()
    print(f"\n[+] Done. Collection '{COLLECTION}' now holds {col.count()} passages.")


if __name__ == "__main__":
    main()
