"""Step 7 - the complete CRAG-RL system, end to end, with an interactive prompt.

Hindi question -> RL query policy (mT5) -> English Wikipedia retrieval (ChromaDB) -> relevance gate
-> Mistral-7B Hindi answer. Prints the query, evidence scores, answer and per-stage latency.

    python run_full_system.py
    python run_full_system.py --policy ./models/mt5_sft_warmstart     # compare with the SFT-only policy
"""
import argparse
import time

import numpy as np

from crag_utils import ABSTAIN_HI, MIN_EVIDENCE_SIM, PPO_DIR, generate_queries, get_device


class CrossLingualRAGPipeline:
    # ---------------------------------------------------------
    # [MODIFIED]: Changed top_k default to 5, lowered min_evidence
    # ---------------------------------------------------------
    def __init__(self, policy_dir=PPO_DIR, top_k=5, min_evidence=MIN_EVIDENCE_SIM, generator=None, retriever=None):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
        from phase2_retriever import KnowledgeRetriever
        from phase3_synthesis import CrossLingualGenerator

        print(f"[*] [1/3] Query policy from {policy_dir} (CPU, keeps GPU memory for Mistral)")
        self.ptok = AutoTokenizer.from_pretrained(policy_dir)
        
        # ---------------------------------------------------------
        # [MODIFIED]: Fix tie_word_embeddings warning
        # ---------------------------------------------------------
        config = AutoConfig.from_pretrained(policy_dir)
        config.tie_word_embeddings = False
        self.policy = AutoModelForSeq2SeqLM.from_pretrained(policy_dir, config=config).to("cpu").eval()
        
        print("[*] [2/3] ChromaDB retriever")
        self.retriever = retriever or KnowledgeRetriever(device=get_device())
        print("[*] [3/3] Mistral-7B reader")
        self.gen = generator or CrossLingualGenerator()
        self.k, self.min_ev = top_k, min_evidence

    def generate_query(self, hindi_prompt):
        return generate_queries(self.policy, self.ptok, [hindi_prompt], "cpu")[0]

    def answer_from_query(self, hindi_prompt, query):
        """Phases 2-3 for a given English query (shared with the baselines in Evaluation.py)."""
        t = time.perf_counter()
        passages = self.retriever.retrieve_with_scores(query, self.k) if query.strip() else []
        evidence = 0.0
        if passages:
            P = self.retriever.encode([hindi_prompt] + [p["text"] for p in passages])
            evidence = float((P[1:] @ P[0]).max())
        t_ret = time.perf_counter() - t
        t = time.perf_counter()
        gated = evidence < self.min_ev
        answer = ABSTAIN_HI if gated else self.gen.generate_answer(hindi_prompt, passages)
        return {"query": query, "passages": passages, "evidence_sim": evidence, "gated": gated, "answer": answer,
                "t_retrieval": t_ret, "t_synthesis": time.perf_counter() - t}

    def query_pipeline(self, hindi_prompt):
        t = time.perf_counter()
        q = self.generate_query(hindi_prompt)
        t_q = time.perf_counter() - t
        out = self.answer_from_query(hindi_prompt, q)
        out.update(hindi_prompt=hindi_prompt, t_query=t_q)
        out["t_total"] = t_q + out["t_retrieval"] + out["t_synthesis"]
        return out


def show(res):
    print(f"\n[Hindi Input] : {res['hindi_prompt']}")
    print(f"[RL Query]    : {res['query']}")
    for i, p in enumerate(res["passages"][:2], 1):
        print(f"[Evidence {i}]  : ({p['score']:.2f}) {p['text'][:130]}...")
    print(f"[Relevance]   : prompt-evidence similarity {res['evidence_sim']:.2f}"
          + ("  -> below threshold, abstaining" if res["gated"] else ""))
    print(f"[Answer]      : {res['answer']}")
    print(f"[Latency]     : query {res['t_query']:.2f}s | retrieval {res['t_retrieval']:.2f}s | "
          f"synthesis {res['t_synthesis']:.2f}s | total {res['t_total']:.2f}s")
    print("-" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=PPO_DIR)
    # ---------------------------------------------------------
    # [MODIFIED]: Changed top_k default to 5 for better synthesis coverage
    # ---------------------------------------------------------
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-evidence", type=float, default=MIN_EVIDENCE_SIM)
    a = ap.parse_args()
    
    pipe = CrossLingualRAGPipeline(a.policy, a.top_k, a.min_evidence)
    print("\nTip on Windows: use Windows Terminal or the VS Code terminal - the old PowerShell console "
          "draws Devanagari incorrectly.\n")
    demo = ["माउंट एवरेस्ट की ऊंचाई कितनी है?", "रूस की राजधानी क्या है?", "दुनिया में कितने महाद्वीप हैं?",
            "च्युइंग गम निगलने से क्या होता है?"]
    print("--- Demo questions (none of these are in the training data) ---")
    times = []
    for q in demo:
        r = pipe.query_pipeline(q)
        show(r)
        times.append(r["t_total"])
    print(f"Mean latency {np.mean(times):.2f}s\n\n--- Interactive mode (type 'exit' to quit) ---")
    while True:
        try:
            q = input("\nप्रश्न दर्ज करें (Enter question): ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if q.lower() in ("exit", "quit", "q"):
            break
        if q:
            try:
                show(pipe.query_pipeline(q))
            except Exception as e:
                print(f"[!] Error: {e}")


if __name__ == "__main__":
    main()