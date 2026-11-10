"""Shared paths, constants and helpers for the CRAG-RL pipeline."""
import json
import os
import random

import numpy as np
import torch

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ------------------------------------------------------------------ paths
DATA_DIR = "./data"
TQA_DIR = "./data/truthfulqa_hindi"
TQA_CSV = os.path.join(TQA_DIR, "TruthfulQA.csv")                       # original English source
TQA_RAW_V0 = os.path.join(TQA_DIR, "truthfulqa_hindi_raw.jsonl")        # v0 (opus-mt), rejected; kept for reference
TQA_FILE = os.path.join(TQA_DIR, "truthfulqa_hindi.jsonl")              # v1 benchmark used for evaluation
TQA_SAMPLE_CSV = os.path.join(TQA_DIR, "truthfulqa_verification_sample.csv")
SFT_TRAIN = os.path.join(DATA_DIR, "sft_train.jsonl")
SFT_VAL = os.path.join(DATA_DIR, "sft_val.jsonl")
CHROMA_DIR = "./data/chroma_db"
COLLECTION = "wiki_passages_v2"
KB_CACHE_DIR = "./data/kb_cache"
SFT_DIR = "./models/mt5_sft_warmstart"
PPO_DIR = "./models/mt5_ppo_aligned"
RESULTS_DIR = "./results"

# ------------------------------------------------------------------ models
POLICY_BASE = "google/mt5-small"
ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
READER_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
MT_HI_EN = "Helsinki-NLP/opus-mt-hi-en"          # Translate-then-Retrieve baseline
TQA_CSV_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"

# ------------------------------------------------------------------ constants
QUERY_PREFIX = "search: "
MAX_PROMPT_TOKENS = 64
MAX_QUERY_TOKENS = 16          # L_max in the reward
ABSTAIN_HI = "दिए गए संदर्भ में इस प्रश्न का उत्तर उपलब्ध नहीं है।"
MIN_EVIDENCE_SIM = 0.25        # below this prompt-passage similarity, the reader abstains without calling Mistral


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(row, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def encode_prompts(tokenizer, prompts, device, max_len=MAX_PROMPT_TOKENS):
    return tokenizer([QUERY_PREFIX + p.strip() for p in prompts], return_tensors="pt", padding=True,
                     truncation=True, max_length=max_len, return_token_type_ids=False).to(device)


@torch.no_grad()
def generate_queries(model, tokenizer, prompts, device, num_beams=2, batch_size=32):
    """Greedy/beam query generation used at inference and evaluation (no sampling)."""
    model.eval()
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = encode_prompts(tokenizer, prompts[i:i + batch_size], device)
        seqs = model.generate(**enc, max_new_tokens=MAX_QUERY_TOKENS, num_beams=num_beams,
                              no_repeat_ngram_size=2, do_sample=False)
        out += [s.strip() for s in tokenizer.batch_decode(seqs, skip_special_tokens=True)]
    return out


def load_tqa_english_refs():
    """English correct/incorrect answers from the original TruthfulQA CSV, keyed by tqa_hi_NNNN."""
    import pandas as pd

    src = TQA_CSV if os.path.exists(TQA_CSV) else TQA_CSV_URL
    df = pd.read_csv(src)
    split = lambda s: [a.strip() for a in str(s).split(";") if a.strip() and a.strip().lower() != "nan"]
    return {f"tqa_hi_{i + 1:04d}": {"best": str(r["Best Answer"]), "correct": split(r["Correct Answers"]),
                                     "incorrect": split(r["Incorrect Answers"])} for i, r in df.iterrows()}
