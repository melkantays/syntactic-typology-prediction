"""embedding_extractor.py
 
Her dilin metnini pretrained bir cok-dilli modelden (XLM-RoBERTa) gecirip,
o dili temsil eden ZENGIN bir vektor (embedding) uretir.
 
Neden: onceki asamada X = 3 elle-yapilmis istatistikti (TTR, ortalama
uzunluk, uzunluk varyansi) ve bu yetmedi (robustness_check.py bunu
dort testle kanitladi). Burada X'i degistiriyoruz: metni, milyonlarca
cumlede egitilmis bir dil modelinin gozunden temsil eden 768 boyutlu
bir vektorle. Hipotez: zengin temsil, yuzey istatistiginin yakalayamadigi
yapisal bilgiyi tasir.
 
Girdi : UD_Corpus/*.txt  (her dil bir dosya)
Cikti : embedding_features.csv  (ISO_Code + emb_0 ... emb_767)
 
NOT: Model egitilmiyor, sadece kullaniliyor (inference). CPU'da calisir,
31 dil icin birkac dakika surer.
"""
 
from __future__ import annotations
 
import argparse
import csv
import sys
from pathlib import Path
 
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
 
MODEL_NAME = "xlm-roberta-base"   # 100 dil kapsar, 768 boyutlu cikti
MAX_TOKENS = 512                  # modelin tek seferde alabilecegi azami uzunluk
N_CHUNKS = 20                     # her dilden kac parca alinacak
 
 
# --- BLOK 1: Modeli yukle -------------------------------------------- #
def load_model(model_name, device):
    """Pretrained tokenizer ve modeli indirir/yukler.
 
    Ilk calistirmada model internetten iner (~1.1 GB), sonrakilerde
    diskten okunur. .eval() modu: model egitim degil, tahmin modunda --
    bu, rastgelelik (dropout) kapatir, sonuclar tekrarlanabilir olur.
    """
    print(f"Model yukleniyor: {model_name} (ilk seferde indirilir, ~1.1 GB)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return tokenizer, model
 
 
# --- BLOK 2: Metni parcalara bol ------------------------------------- #
def make_chunks(text, tokenizer, max_tokens, n_chunks):
    """Uzun metni, modelin alabilecegi boyutta parcalara boler.
 
    Model tek seferde en fazla 512 token isleyebilir. Metnin tamamini
    veremeyiz; bu yuzden metni esit araliklarla ornekleyip n_chunks
    kadar parca aliyoruz. Esit aralik onemli: metnin sadece basindan
    almak, o dilin bir bolumune (ornegin giris cumlelerine) yanlilik
    yaratirdi.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    usable = max_tokens - 2          # [CLS] ve [SEP] icin yer birak
    total = len(token_ids)
 
    if total <= usable:
        return [token_ids]
 
    # Metin boyunca esit araliklarla baslangic noktalari sec
    starts = np.linspace(0, total - usable, num=min(n_chunks, total // usable + 1))
    chunks = []
    for s in starts:
        s = int(s)
        chunks.append(token_ids[s:s + usable])
    return chunks
 
 
# --- BLOK 3: Bir dilin vektorunu hesapla ----------------------------- #
def embed_language(text, tokenizer, model, device):
    """Metni modelden gecirip tek bir 768 boyutlu vektor uretir.
 
    Iki asamali ortalama:
      1) Her parca icin, token vektorlerinin ortalamasi alinir
         (mean pooling). Dolgu (padding) tokenlari maskeyle disarida
         birakilir -- yoksa bos yerler ortalamayi bozar.
      2) Tum parcalarin vektorleri ortalanir -> dilin tek temsili.
 
    torch.no_grad(): gradyan hesabi kapali. Egitim yapmiyoruz, sadece
    tahmin; bu, hem hizlandirir hem bellek tasarrufu saglar.
    """
    chunks = make_chunks(text, tokenizer, MAX_TOKENS, N_CHUNKS)
    vectors = []
 
    with torch.no_grad():
        for chunk in chunks:
            ids = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
            input_ids = torch.tensor([ids], device=device)
            attention = torch.ones_like(input_ids)
 
            output = model(input_ids=input_ids, attention_mask=attention)
            hidden = output.last_hidden_state          # (1, uzunluk, 768)
 
            # Mean pooling: maskeyle agirlikli ortalama
            mask = attention.unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            vectors.append((summed / counts).squeeze(0).cpu().numpy())
 
    return np.mean(vectors, axis=0)
 
 
# --- BLOK 4: Tum korpusu isle ---------------------------------------- #
def process_corpus(corpus_dir, tokenizer, model, device):
    """UD_Corpus icindeki her .txt dosyasi icin bir vektor uretir."""
    rows = []
    files = sorted(corpus_dir.glob("*.txt"))
    print(f"{len(files)} dil bulundu.\n")
 
    for i, path in enumerate(files, 1):
        iso_code = path.stem
        text = path.read_text(encoding="utf-8", errors="replace").strip()
 
        if not text:
            print(f"[{i}/{len(files)}] {iso_code}: BOS dosya, atlandi.")
            continue
 
        vector = embed_language(text, tokenizer, model, device)
        rows.append((iso_code, vector))
        print(f"[{i}/{len(files)}] {iso_code}: tamam "
              f"({len(vector)} boyut)")
 
    return rows
 
 
# --- BLOK 5: Kaydet -------------------------------------------------- #
def export(rows, out_path):
    """ISO_Code + emb_0..emb_N seklinde CSV yazar."""
    if not rows:
        raise ValueError("Hicbir dil islenemedi.")
    dim = len(rows[0][1])
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ISO_Code"] + [f"emb_{i}" for i in range(dim)])
        for iso_code, vector in rows:
            writer.writerow([iso_code] + [f"{v:.6f}" for v in vector])
 
 
def main(argv=None):
    p = argparse.ArgumentParser(description="Pretrained dil temsili cikarici.")
    p.add_argument("--corpus", type=Path, default=Path("UD_Corpus"))
    p.add_argument("--out", type=Path, default=Path("embedding_features.csv"))
    p.add_argument("--model", default=MODEL_NAME)
    args = p.parse_args(argv)
 
    if not args.corpus.is_dir():
        print(f"HATA: {args.corpus} klasoru yok.", file=sys.stderr)
        return 1
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device}")
 
    tokenizer, model = load_model(args.model, device)
    rows = process_corpus(args.corpus, tokenizer, model, device)
    export(rows, args.out)
 
    print(f"\n[TAMAM] {len(rows)} dil -> {args.out}")
    print(f"Her dil {len(rows[0][1])} boyutlu bir vektorle temsil edildi.")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 