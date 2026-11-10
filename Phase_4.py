"""Step 6 - Phase 1b: PPO alignment of the query generator (proposal Sec. 3.2, Eqs. 1-3).

Fixes over the first version:
  * Real PPO (was REINFORCE): clipped surrogate, value head + GAE, entropy bonus, KL to the frozen SFT
    policy, several PPO epochs per batch of rollouts.
  * Log-probabilities scored with the correct decoder inputs (passing labels= made the model shift the
    sequence a second time, so every token was scored after a pad instead of the previous word).
  * The reward now depends on the Hindi question. The old reward only asked "does this query match some
    passage?", so a generic query like "japan capital city tokyo" scored high for every prompt. It also
    embedded queries with ChromaDB's default English model, not the multilingual encoder of the index.

Reward (proposal Eq. 1, see docs/DESIGN_NOTES.md for the alpha-term proxy):
    R = alpha * max_j cos(x_hi, c_j)        evidence supports the *question* (cross-lingual)
      + beta  * mean_j cos(q, c_j)          query lands in a coherent evidence region (Eq. 1 CosineSim)
      + delta * cos(x_hi, q)                query means what the question asks
      - gamma * min(1, |q| / L_max)         brevity
    Empty / degenerate / repetitive queries get -1.

    python Phase_4.py                       # ~20-40 min on a laptop GPU
    python Phase_4.py --beta 0 --out ./models/ppo_no_cos      # reward ablations
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from crag_utils import (MAX_QUERY_TOKENS, PPO_DIR, SFT_DIR, SFT_VAL, SFT_TRAIN, encode_prompts, generate_queries,
                        get_device, read_jsonl, set_seed)


# ------------------------------------------------------------------ reward
class CrossLingualRewardEngine:
    def __init__(self, retriever, tokenizer, alpha=1.0, beta=0.3, delta=0.5, gamma=0.2, top_k=3,
                 lmax=MAX_QUERY_TOKENS):
        self.r, self.tok = retriever, tokenizer
        self.alpha, self.beta, self.delta, self.gamma = alpha, beta, delta, gamma
        self.k, self.lmax = top_k, lmax
        self.cache = {}

    def _degenerate(self, q):
        toks = q.lower().split()
        return len(toks) < 2 or len(set(toks)) / len(toks) < 0.7

    def __call__(self, prompts, queries):
        todo = [(p, q) for p, q in zip(prompts, queries) if (p, q) not in self.cache and not self._degenerate(q)]
        if todo:
            ps, qs = [p for p, _ in todo], [q for _, q in todo]
            P = self.r.encode(ps)
            Q = self.r.encode(qs)
            hits = self.r.search(qs, self.k, query_emb=Q)
            E = self.r.encode([h["text"] for hs in hits for h in hs]).reshape(len(todo), self.k, -1)
            for i, (p, q) in enumerate(todo):
                self.cache[(p, q)] = {"evidence": float((E[i] @ P[i]).max()),
                                      "qc": float(np.mean([h["score"] for h in hits[i]])),
                                      "qrel": float(P[i] @ Q[i])}
        rewards, infos = [], []
        for p, q in zip(prompts, queries):
            if self._degenerate(q):
                rewards.append(-1.0)
                infos.append({"evidence": 0.0, "qc": 0.0, "qrel": 0.0, "len": 0.0, "degenerate": 1})
                continue
            info = dict(self.cache[(p, q)])
            info["len"] = min(1.0, len(self.tok(q, add_special_tokens=False)["input_ids"]) / self.lmax)
            info["degenerate"] = 0
            rewards.append(self.alpha * info["evidence"] + self.beta * info["qc"] + self.delta * info["qrel"]
                           - self.gamma * info["len"])
            infos.append(info)
        return rewards, infos


# ------------------------------------------------------------------ policy + value head
class PolicyWithValue(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.v_head = nn.Linear(model.config.d_model, 1)
        nn.init.normal_(self.v_head.weight, std=0.01)
        nn.init.zeros_(self.v_head.bias)

    def forward_seq(self, enc, seqs, with_value=True):
        """Teacher-forced scores of seqs[:, 1:] given decoder inputs seqs[:, :-1]."""
        out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                         decoder_input_ids=seqs[:, :-1], output_hidden_states=with_value, return_dict=True)
        logp_all = torch.log_softmax(out.logits.float(), -1)
        tgt = seqs[:, 1:]
        logp = logp_all.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        entropy = -(logp_all.exp() * logp_all).sum(-1)
        values = self.v_head(out.decoder_hidden_states[-1].float()).squeeze(-1) if with_value else None
        return logp, entropy, values


def response_mask(seqs, pad_id, eos_id):
    tgt = seqs[:, 1:]
    is_eos = (tgt == eos_id).int()
    after_eos = (is_eos.cumsum(1) - is_eos) > 0
    return ((~after_eos) & ((tgt != pad_id) | (tgt == eos_id))).float()


def masked_mean(x, m):
    return (x * m).sum() / m.sum().clamp(min=1.0)


def whiten(x, m):
    mu = masked_mean(x, m)
    var = masked_mean((x - mu) ** 2, m)
    return (x - mu) * torch.rsqrt(var + 1e-8) * m


def gae(token_r, values, mask, gamma=1.0, lam=0.95):
    T = token_r.shape[1]
    adv = torch.zeros_like(token_r)
    last = torch.zeros(token_r.shape[0], device=token_r.device)
    for t in reversed(range(T)):
        nxt_v = values[:, t + 1] * mask[:, t + 1] if t + 1 < T else torch.zeros_like(last)
        nxt_m = mask[:, t + 1] if t + 1 < T else torch.zeros_like(last)
        delta = token_r[:, t] + gamma * nxt_v - values[:, t]
        last = delta + gamma * lam * nxt_m * last
        adv[:, t] = last * mask[:, t]
    return adv, (adv + values) * mask


class RunningStats:
    def __init__(self):
        self.mean, self.var, self.n = 0.0, 1.0, 1e-4

    def update(self, xs):
        xs = np.asarray(xs, float)
        bm, bv, bn = xs.mean(), xs.var(), len(xs)
        d, tot = bm - self.mean, self.n + bn
        self.mean += d * bn / tot
        self.var = (self.var * self.n + bv * bn + d * d * self.n * bn / tot) / tot
        self.n = tot

    @property
    def std(self):
        return float(np.sqrt(self.var) + 1e-8)


# ------------------------------------------------------------------ PPO
class PPOTrainer:
    def __init__(self, policy, ref, tok, reward_fn, a, device):
        self.pol, self.ref, self.tok, self.rf, self.a, self.dev = policy, ref, tok, reward_fn, a, device
        params = [p for p in policy.parameters() if p.requires_grad]
        self.params = params
        self.opt = torch.optim.AdamW(params, lr=a.lr, eps=1e-6)
        self.stats = RunningStats()

    @torch.no_grad()
    def rollout(self, prompts):
        self.pol.eval()
        enc = encode_prompts(self.tok, prompts, self.dev)
        seqs = self.pol.model.generate(**enc, max_new_tokens=MAX_QUERY_TOKENS, do_sample=True,
                                       temperature=self.a.temperature, top_k=self.a.top_k)
        queries = [s.strip() for s in self.tok.batch_decode(seqs, skip_special_tokens=True)]
        t0 = time.perf_counter()
        rewards, infos = self.rf(prompts, queries)
        rtime = time.perf_counter() - t0
        mask = response_mask(seqs, self.tok.pad_token_id, self.tok.eos_token_id)
        old_logp, _, old_v = self.pol.forward_seq(enc, seqs)
        ref_logp, _, _ = self.ref.forward_seq(enc, seqs, with_value=False)
        self.stats.update(rewards)
        scaled = (torch.tensor(rewards, device=self.dev) - self.stats.mean) / self.stats.std
        token_r = torch.zeros_like(old_logp)
        last = (mask.cumsum(1) * mask).argmax(1)
        token_r[torch.arange(len(prompts)), last] = scaled
        old_v = old_v * mask
        adv, ret = gae(token_r, old_v, mask, 1.0, self.a.lam)
        adv = whiten(adv, mask)
        return dict(enc=enc, seqs=seqs, mask=mask, old_logp=old_logp, ref_logp=ref_logp.float(), old_v=old_v,
                    adv=adv, ret=ret, rewards=rewards, infos=infos, queries=queries, rtime=rtime)

    def update(self, b):
        a, n = self.a, len(b["rewards"])
        st = {k: [] for k in ("pg", "vf", "ent", "kl", "clip")}
        self.pol.eval()  # dropout off: pi_old == pi_theta at the first PPO epoch (ratio starts at exactly 1)
        for _ in range(a.ppo_epochs):
            perm = torch.randperm(n)
            for s in range(0, n, a.minibatch):
                i = perm[s:s + a.minibatch]
                enc = {k: v[i] for k, v in b["enc"].items()}
                seqs, m = b["seqs"][i], b["mask"][i]
                logp, ent, v = self.pol.forward_seq(enc, seqs)
                ratio = torch.exp(logp - b["old_logp"][i])
                adv = b["adv"][i]
                pg = masked_mean(torch.max(-adv * ratio, -adv * ratio.clamp(1 - a.clip, 1 + a.clip)), m)
                v_clip = b["old_v"][i] + (v - b["old_v"][i]).clamp(-a.clip, a.clip)
                vf = 0.5 * masked_mean(torch.max((v - b["ret"][i]) ** 2, (v_clip - b["ret"][i]) ** 2), m)
                e = masked_mean(ent, m)
                lr_ = (b["ref_logp"][i] - logp).clamp(-20, 20)
                kl = masked_mean(torch.exp(lr_) - lr_ - 1.0, m)   # k3 estimator of KL(pi || pi_SFT)
                loss = pg + a.vf_coef * vf - a.ent_coef * e + a.kl_coef * kl
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, 1.0)
                self.opt.step()
                with torch.no_grad():
                    st["pg"].append(pg.item()); st["vf"].append(vf.item()); st["ent"].append(e.item())
                    st["kl"].append(kl.item())
                    st["clip"].append(masked_mean(((ratio - 1).abs() > a.clip).float(), m).item())
        return {k: float(np.mean(x)) for k, x in st.items()}

    def evaluate(self, prompts):
        qs = generate_queries(self.pol.model, self.tok, prompts, self.dev)
        r, inf = self.rf(prompts, qs)
        return float(np.mean(r)), float(np.mean([x["evidence"] for x in inf])), list(zip(prompts[:4], qs[:4]))


def save_policy(policy, tok, path):
    os.makedirs(path, exist_ok=True)
    policy.model.save_pretrained(path)
    tok.save_pretrained(path)
    torch.save(policy.v_head.state_dict(), os.path.join(path, "value_head.pt"))


def plot(log, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    keys = [("reward", "Reward"), ("evidence", "Prompt-evidence similarity"), ("qrel", "Prompt-query similarity"),
            ("len", "Normalised query length"), ("kl", "KL(pi || pi_SFT)"), ("ent", "Entropy")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    steps = [r["step"] for r in log]
    for ax, (k, t) in zip(axes.flat, keys):
        ax.plot(steps, [r[k] for r in log], alpha=0.8, label="train batch")
        if k == "reward":
            pts = [(r["step"], r["val_reward"]) for r in log if "val_reward" in r]
            if pts:
                ax.plot(*zip(*pts), "o--", label="held-out (greedy)")
            ax.legend(fontsize=8)
        ax.set_title(t); ax.set_xlabel("PPO step"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150)
    print(f"[+] Training curves -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=SFT_DIR)
    ap.add_argument("--out", default=PPO_DIR)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--minibatch", type=int, default=8)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--vf-coef", type=float, default=0.1)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.2)
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--train-embeddings", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    set_seed(a.seed)
    device = get_device()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from phase2_retriever import KnowledgeRetriever

    tok = AutoTokenizer.from_pretrained(a.sft)
    policy = PolicyWithValue(AutoModelForSeq2SeqLM.from_pretrained(a.sft)).to(device)
    if not a.train_embeddings:
        policy.model.shared.weight.requires_grad_(False)
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    if device == "cuda" and torch.cuda.is_bf16_supported():
        ref = ref.to(torch.bfloat16)  # frozen reference only: saves memory

    retriever = KnowledgeRetriever(device=device)
    rf = CrossLingualRewardEngine(retriever, tok, a.alpha, a.beta, a.delta, a.gamma)
    train_prompts = [r["hindi_prompt"] for r in read_jsonl(SFT_TRAIN)]
    val_prompts = [r["hindi_prompt"] for r in read_jsonl(SFT_VAL)][: a.n_val]
    print(f"[*] PPO on {device} | KB={retriever.count()} passages | {len(train_prompts)} train prompts | "
          f"{len(val_prompts)} held-out | reward a={a.alpha} b={a.beta} d={a.delta} g={a.gamma}")

    trainer = PPOTrainer(policy, ref, tok, rf, a, device)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "ppo_config.json"), "w"), indent=2)
    log_path = os.path.join(a.out, "train_log.jsonl")
    open(log_path, "w").close()

    v0, ev0, _ = trainer.evaluate(val_prompts)
    print(f"[*] step 0 (SFT policy) held-out reward {v0:+.3f}, evidence {ev0:.3f}")
    best = v0
    save_policy(policy, tok, a.out)
    rng = random.Random(a.seed)
    log = []
    for step in range(1, a.steps + 1):
        t0 = time.perf_counter()
        b = trainer.rollout(rng.sample(train_prompts, min(a.batch, len(train_prompts))))
        st = trainer.update(b)
        inf = b["infos"]
        rec = {"step": step, "reward": float(np.mean(b["rewards"])),
               **{k: float(np.mean([x[k] for x in inf])) for k in ("evidence", "qc", "qrel", "len", "degenerate")},
               **st, "sec": round(time.perf_counter() - t0, 2), "sample_query": b["queries"][0]}
        if step % a.eval_every == 0 or step == a.steps:
            vr, vev, ex = trainer.evaluate(val_prompts)
            rec.update(val_reward=vr, val_evidence=vev, val_examples=ex)
            if vr > best:
                best = vr
                save_policy(policy, tok, a.out)
                rec["saved"] = True
        log.append(rec)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"step {step:3d} | R {rec['reward']:+.3f} ev {rec['evidence']:.3f} qrel {rec['qrel']:.3f} "
              f"len {rec['len']:.2f} | kl {st['kl']:.4f} clip {st['clip']:.2f} ent {st['ent']:.2f}"
              + (f" | held-out R {rec['val_reward']:+.3f}{' *saved*' if rec.get('saved') else ''}" if "val_reward" in rec else "")
              + f" | q: {rec['sample_query'][:45]}")

    print(f"\n[+] Best held-out reward {best:+.3f} (SFT start {v0:+.3f}). Policy saved to {a.out}")
    plot(log, os.path.join(a.out, "ppo_curves.png"))
    tok2 = AutoTokenizer.from_pretrained(a.out)
    best_model = AutoModelForSeq2SeqLM.from_pretrained(a.out).to(device)
    probes = ["रूस की राजधानी क्या है?", "दुनिया में कितने महाद्वीप हैं?", "माउंट एवरेस्ट की ऊंचाई कितनी है?",
              "जापान की राजधानी क्या है?"]
    for p, q in zip(probes, generate_queries(best_model, tok2, probes, device)):
        print(f"  {p}  ->  {q}")


if __name__ == "__main__":
    main()
