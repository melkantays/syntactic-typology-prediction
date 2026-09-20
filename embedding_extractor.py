"""embedding_extractor.py
 
Passes each language's text through a pretrained multilingual model
(XLM-RoBERTa) to produce a RICH vector (embedding) representing that
language.
 
Rationale: in the previous stage X consisted of 3 hand-built statistics
(TTR, mean length, length variance), and that was not enough --
robustness_check.py established this across four tests. Here we replace
X with a 768-dimensional vector that represents the text through the
eyes of a model trained on millions of sentences. Hypothesis: a rich
representation carries structural information that surface statistics
cannot capture.
 
Input  : UD_Corpus/*.txt  (one file per language)
Output : embedding_features.csv  (ISO_Code + emb_0 ... emb_767)
 
NOTE: the model is not trained here, only used (inference). Runs on CPU;
a few minutes for 31 languages.
"""
 
from __future__ import annotations
 
import argparse
import csv
import sys
from pathlib import Path
 
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
 
MODEL_NAME = "xlm-roberta-base"   # covers 100 languages, 768-dim output
MAX_TOKENS = 512                  # the model's maximum input length
N_CHUNKS = 20                     # how many chunks to sample per language
 
 
# --- BLOCK 1: load the model ------------------------------------------ #
def load_model(model_name, device):
    """Download / load the pretrained tokenizer and model.
 
    On the first run the model is downloaded (~1.1 GB); afterwards it is
    read from disk. .eval() puts the model in inference rather than
    training mode, which disables dropout and therefore makes the
    outputs deterministic and reproducible.
    """
    print(f"Loading model: {model_name} (downloaded on first run, ~1.1 GB)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return tokenizer, model
 
 
# --- BLOCK 2: split the text into chunks ------------------------------ #
def make_chunks(text, tokenizer, max_tokens, n_chunks):
    """Split long text into pieces the model can accept.
 
    The model processes at most 512 tokens at a time, so the full text
    cannot be passed in one go. Instead we sample n_chunks pieces at
    evenly spaced positions. Even spacing matters: taking only the head
    of the file would bias the representation toward one section of the
    corpus (for instance, opening sentences).
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    usable = max_tokens - 2          # leave room for [CLS] and [SEP]
    total = len(token_ids)
 
    if total <= usable:
        return [token_ids]
 
    # Pick evenly spaced starting points across the whole text.
    starts = np.linspace(0, total - usable, num=min(n_chunks, total // usable + 1))
    chunks = []
    for s in starts:
        s = int(s)
        chunks.append(token_ids[s:s + usable])
    return chunks
 
 
# --- BLOCK 3: compute one language's vector --------------------------- #
def embed_language(text, tokenizer, model, device):
    """Run the text through the model and return a single 768-dim vector.
 
    Averaging happens twice:
      1) Within each chunk, the token vectors are averaged (mean
         pooling). Padding tokens are excluded via the attention mask --
         otherwise empty positions would distort the mean.
      2) The chunk vectors are then averaged into one representation of
         the language.
 
    torch.no_grad(): gradient tracking is off. We are not training, only
    predicting, which makes this both faster and far lighter on memory.
    """
    chunks = make_chunks(text, tokenizer, MAX_TOKENS, N_CHUNKS)
    vectors = []
 
    with torch.no_grad():
        for chunk in chunks:
            ids = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
            input_ids = torch.tensor([ids], device=device)
            attention = torch.ones_like(input_ids)
 
            output = model(input_ids=input_ids, attention_mask=attention)
            hidden = output.last_hidden_state          # (1, length, 768)
 
            # Mean pooling: mask-weighted average
            mask = attention.unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            vectors.append((summed / counts).squeeze(0).cpu().numpy())
 
    return np.mean(vectors, axis=0)
 
 
# --- BLOCK 4: process the whole corpus -------------------------------- #
def process_corpus(corpus_dir, tokenizer, model, device):
    """Produce one vector for every .txt file in the corpus directory."""
    rows = []
    files = sorted(corpus_dir.glob("*.txt"))
    print(f"{len(files)} language file(s) found.\n")
 
    for i, path in enumerate(files, 1):
        iso_code = path.stem
        text = path.read_text(encoding="utf-8", errors="replace").strip()
 
        if not text:
            print(f"[{i}/{len(files)}] {iso_code}: EMPTY file, skipped.")
            continue
 
        vector = embed_language(text, tokenizer, model, device)
        rows.append((iso_code, vector))
        print(f"[{i}/{len(files)}] {iso_code}: done "
              f"({len(vector)} dimensions)")
 
    return rows
 
 
# --- BLOCK 5: save ---------------------------------------------------- #
def export(rows, out_path):
    """Write a CSV of ISO_Code + emb_0..emb_N."""
    if not rows:
        raise ValueError("No language could be processed.")
    dim = len(rows[0][1])
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ISO_Code"] + [f"emb_{i}" for i in range(dim)])
        for iso_code, vector in rows:
            writer.writerow([iso_code] + [f"{v:.6f}" for v in vector])
 
 
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Pretrained language-representation extractor."
    )
    p.add_argument("--corpus", type=Path, default=Path("UD_Corpus"))
    p.add_argument("--out", type=Path, default=Path("embedding_features.csv"))
    p.add_argument("--model", default=MODEL_NAME)
    args = p.parse_args(argv)
 
    if not args.corpus.is_dir():
        print(f"ERROR: directory {args.corpus} does not exist.", file=sys.stderr)
        return 1
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
 
    tokenizer, model = load_model(args.model, device)
    rows = process_corpus(args.corpus, tokenizer, model, device)
    export(rows, args.out)
 
    print(f"\n[DONE] {len(rows)} language(s) -> {args.out}")
    print(f"Each language is represented by a {len(rows[0][1])}-dim vector.")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 