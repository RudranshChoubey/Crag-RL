"""Phase 3 - frozen Mistral-7B-Instruct writes the Hindi answer from English evidence.

One generator class used by run_full_system.py and Evaluation.py, so every system is answered with
the same prompt and decoding settings. Changes:
  * Few-shot prompt with one answerable AND one unanswerable example.
  * Explicit "Hindi only, no English explanations"; English text in brackets is stripped.
  * Relevance gate: if no passage is about the question (prompt-passage similarity < MIN_EVIDENCE_SIM)
    the system abstains instead of answering from unrelated text (the old demo answered "5,030 m"
    for Everest from the generic "Mountain" article).
  * A no-context mode (No-Retrieval baseline) and a query-rewrite mode (LLM-Rewrite baseline).
"""
import re

import torch

from crag_utils import ABSTAIN_HI, READER_MODEL_ID, get_device

RAG_INSTRUCTIONS = (
    "You are a careful, factual assistant. Answer the Hindi question using ONLY the English context passages.\n"
    "Rules:\n"
    "1. Use only facts stated in the context. Do not use outside knowledge.\n"
    "2. Write the answer in simple, natural Hindi (Devanagari), 1-2 short sentences.\n"
    "3. Do not add English explanations, notes or translations.\n"
    f"4. If the context does not answer the question, reply exactly: {ABSTAIN_HI}"
)
SHOT_1_USER = ("Context:\n[1] Mount Everest is Earth's highest mountain above sea level, located in the Himalayas. "
               "Its elevation was most recently established in 2020 as 8,848.86 metres.\n\n"
               "Question: माउंट एवरेस्ट की ऊंचाई कितनी है?")
SHOT_1_ANS = "माउंट एवरेस्ट की ऊंचाई 8,848.86 मीटर है।"
SHOT_2_USER = ("Context:\n[1] Tokyo is the capital and largest city of Japan. It is on the island of Honshu.\n\n"
               "Question: ब्राज़ील की राजधानी क्या है?")
SHOT_2_ANS = ABSTAIN_HI
NORAG_INSTRUCTIONS = ("Answer the Hindi question truthfully in simple Hindi (Devanagari), 1-2 short sentences. "
                      "Do not add English explanations.")
REWRITE_INSTRUCTIONS = ("Convert the Hindi question into a short English search query for Wikipedia: 3-8 keywords, "
                        "key entities first, no full sentence, no explanation. Output only the query.")

_PAREN_EN = re.compile(r"\s*[\(\[][^\)\]]*[A-Za-z][^\)\]]*[\)\]]")


def clean_answer(text):
    text = text.strip().strip('"').strip()
    if "उत्तर उपलब्ध नहीं" in text or "पर्याप्त जानकारी नहीं" in text:
        return ABSTAIN_HI
    text = _PAREN_EN.sub("", text)
    lines = [l for l in text.splitlines() if l.strip()]
    hindi_lines = [l for l in lines if re.search(r"[ऀ-ॿ]", l)]  # drop English-only lines
    text = " ".join(hindi_lines or lines).strip()
    parts = re.split(r"(?<=[।\.\?\!])\s+", text)
    return " ".join(parts[:2]).strip()


def format_context(passages, max_chars=2400):
    out, used = [], 0
    for i, p in enumerate(passages, 1):
        t = p["text"] if isinstance(p, dict) else p
        s = f"[{i}] {t}"[: max(0, max_chars - used)]
        if s:
            out.append(s)
            used += len(s)
    return "\n\n".join(out)


class CrossLingualGenerator:
    def __init__(self, model_id=READER_MODEL_ID, load_4bit=True):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.device == "cuda" and load_4bit:
            from transformers import BitsAndBytesConfig
            print(f"[*] Loading {model_id} in 4-bit on GPU ...")
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=q, device_map={"": 0})
        else:
            print(f"[!] Loading {model_id} on {self.device} without quantisation (slow, needs ~16 GB RAM).")
            dtype = torch.float16 if self.device == "mps" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def chat(self, messages, max_new_tokens=120):
        enc = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                                 return_dict=True)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def generate_answer(self, hindi_prompt, passages=None, max_new_tokens=120):
        """passages=None -> No-Retrieval baseline (model answers from its own knowledge)."""
        if passages is None:
            msgs = [{"role": "user", "content": f"{NORAG_INSTRUCTIONS}\n\nप्रश्न: {hindi_prompt}"}]
        else:
            if not passages:
                return ABSTAIN_HI
            msgs = [{"role": "user", "content": f"{RAG_INSTRUCTIONS}\n\n{SHOT_1_USER}"},
                    {"role": "assistant", "content": SHOT_1_ANS},
                    {"role": "user", "content": SHOT_2_USER},
                    {"role": "assistant", "content": SHOT_2_ANS},
                    {"role": "user", "content": f"Context:\n{format_context(passages)}\n\nQuestion: {hindi_prompt}"}]
        return clean_answer(self.chat(msgs, max_new_tokens))

    def rewrite_query(self, hindi_prompt):
        out = self.chat([{"role": "user", "content": f"{REWRITE_INSTRUCTIONS}\n\n{hindi_prompt}"}], 24)
        first = out.strip().splitlines()[0] if out.strip() else ""
        return first.strip().strip('"').strip("`")[:120]


if __name__ == "__main__":
    from phase2_retriever import KnowledgeRetriever

    retriever = KnowledgeRetriever()
    gen = CrossLingualGenerator()
    for prompt, query in [("माउंट एवरेस्ट की ऊंचाई कितनी है?", "Mount Everest elevation height"),
                          ("रूस की राजधानी क्या है?", "Russia capital city")]:
        ps = retriever.retrieve_with_scores(query, top_k=3)
        print(f"\nQ: {prompt}\n  top passage ({ps[0]['score']:.3f}): {ps[0]['text'][:140]}...")
        print(f"  answer: {gen.generate_answer(prompt, ps)}")
    print(f"\nNo-retrieval: {gen.generate_answer('रूस की राजधानी क्या है?', None)}")
