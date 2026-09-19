"""embedding_compiler.py
 
768 boyutlu dil temsillerini PCA ile 10 boyuta indirir ve ayni 7 URIEL
hedefiyle birlestirerek ikinci deneyin egitim matrisini uretir.
 
Neden PCA: 768 ozellik ama sadece ~31 dil var. Bu oranda model gercek
oruntu yerine GURULTUYU ezberler (boyut laneti / curse of dimensionality).
768'i, en cok bilgi tasiyan 10 eksene sikistirinca zengin temsilin ozu
korunur ama veri/boyut orani makul olur.
 
Neden ayni dil kumesi: yuzey deneyi (3 istatistik) ile bu deneyi
karsilastiracagiz. Farkli dil kumelerinde kosarsak, sonuc farkinin
ozellikten mi yoksa dil kumesinden mi geldigini ayiramayiz. Bu yuzden
training_matrix.csv'deki dillerle KESISTIRIYORUZ -- bu ayni zamanda
ara/jpn/vie gibi icerigi bozuk dosyalari da otomatik eler.
 
Girdi : embedding_features.csv  (ISO_Code + emb_0..emb_767)
        training_matrix.csv     (yuzey deneyinin dil kumesi icin)
Cikti : training_matrix_emb.csv (ISO_Code + pca_0..pca_9 + 7 hedef)
"""
 
from __future__ import annotations
 
import argparse
import sys
from pathlib import Path
 
import lang2vec.lang2vec as l2v
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
 
TARGET_FEATURES = [
    "S_SOV", "S_SVO", "S_VSO",
    "S_ADPOSITION_BEFORE_NOUN", "S_ADPOSITION_AFTER_NOUN",
    "S_CASE_SUFFIX", "S_ADJECTIVE_BEFORE_NOUN",
]
 
 
# --- BLOK 1: Embedding dosyasini oku --------------------------------- #
def load_embeddings(path):
    """embedding_features.csv'yi okur; ISO kodlari ve 768 boyutlu matris."""
    frame = pd.read_csv(path)
    emb_cols = [c for c in frame.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("emb_ ile baslayan sutun yok.")
    return frame["ISO_Code"].tolist(), frame[emb_cols].to_numpy(np.float64)
 
 
# --- BLOK 2: Ayni dil kumesine kisitla ------------------------------- #
def restrict_to_reference(iso_codes, matrix, reference_path):
    """Yuzey deneyindeki dillerle kesistirir.
 
    Iki isi birden yapar:
      1) Adil karsilastirma: iki deney ayni dillerde kosar.
      2) Temizlik: icerigi bozuk dosyalar (ornegin sadece tire iceren
         ara.txt) yuzey deneyinde zaten elenmisti; burada da otomatik
         disarida kalir -- anlamsiz vektorler veriye karismaz.
    """
    if not Path(reference_path).exists():
        print(f"UYARI: {reference_path} yok, kisitlama yapilmadi.")
        return iso_codes, matrix
 
    ref = pd.read_csv(reference_path)
    ref_langs = set(ref["ISO_Code"].tolist())
 
    keep = [i for i, code in enumerate(iso_codes) if code in ref_langs]
    dropped = [c for c in iso_codes if c not in ref_langs]
    if dropped:
        print(f"Referans kumede olmayan {len(dropped)} dil elendi: {dropped}")
 
    return [iso_codes[i] for i in keep], matrix[keep]
 
 
# --- BLOK 3: PCA ile boyut indirgeme --------------------------------- #
def reduce_dimensions(matrix, n_components):
    """768 boyutu n_components boyuta indirir.
 
    Once StandardScaler: her boyut farkli olcekte olabilir, PCA olcege
    duyarlidir. Sonra PCA: veriyi en cok degiskenlik (varyans) tasiyan
    eksenlere yeniden yazar; ilk eksenler en cok bilgiyi tasir.
 
    NOT (durustluk): PCA burada TUM X uzerinde egitiliyor. PCA
    denetimsizdir -- hedefleri (Y) hic gormez -- bu yuzden klasik veri
    sizintisi olusturmaz. Yine de en titiz yontem PCA'yi capraz
    dogrulama katlamalarinin ICINDE calistirmaktir; makalede bu tercih
    belirtilmelidir.
    """
    scaled = StandardScaler().fit_transform(matrix)
    n_components = min(n_components, scaled.shape[0], scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(scaled)
 
    explained = pca.explained_variance_ratio_
    print(f"\nPCA: {matrix.shape[1]} boyut -> {n_components} boyut")
    print(f"Korunan bilgi (toplam varyans): {explained.sum():.1%}")
    print("Ilk 5 eksenin payi: " +
          ", ".join(f"{v:.1%}" for v in explained[:5]))
    return reduced, explained
 
 
# --- BLOK 4: URIEL hedeflerini cek ----------------------------------- #
def fetch_targets(langs):
    """Ayni 7 gozlenmis (wals) hedefi ceker; eksigi olan dili eler."""
    data = l2v.get_features(langs, "syntax_wals", header=True)
    index = {name: i for i, name in enumerate(data["CODE"])}
 
    rows, dropped = [], []
    for lang in langs:
        if lang not in data:
            dropped.append(lang)
            continue
        vector = data[lang]
        row, complete = {"ISO_Code": lang}, True
        for feat in TARGET_FEATURES:
            value = vector[index[feat]]
            if value == "--":
                complete = False
                break
            row[feat] = float(value)
        (rows if complete else dropped).append(row if complete else lang)
 
    if dropped:
        print(f"Eksik hedef verisi yuzunden elenen: {dropped}")
    return pd.DataFrame(rows)
 
 
def main(argv=None):
    p = argparse.ArgumentParser(description="Embedding + PCA matris derleyici.")
    p.add_argument("--emb", type=Path, default=Path("embedding_features.csv"))
    p.add_argument("--reference", type=Path, default=Path("training_matrix.csv"))
    p.add_argument("--out", type=Path, default=Path("training_matrix_emb.csv"))
    p.add_argument("--components", type=int, default=10)
    args = p.parse_args(argv)
 
    if not args.emb.exists():
        print(f"HATA: {args.emb} yok.", file=sys.stderr)
        return 1
 
    iso_codes, matrix = load_embeddings(args.emb)
    print(f"Okunan: {len(iso_codes)} dil, {matrix.shape[1]} boyut")
 
    iso_codes, matrix = restrict_to_reference(iso_codes, matrix, args.reference)
    print(f"Kisitlamadan sonra: {len(iso_codes)} dil")
 
    reduced, _ = reduce_dimensions(matrix, args.components)
 
    x_frame = pd.DataFrame(
        reduced, columns=[f"pca_{i}" for i in range(reduced.shape[1])]
    )
    x_frame.insert(0, "ISO_Code", iso_codes)
 
    print(f"\n[{len(iso_codes)} dil] URIEL'den {len(TARGET_FEATURES)} hedef...")
    y_frame = fetch_targets(iso_codes)
 
    final = pd.merge(x_frame, y_frame, on="ISO_Code")
    final.to_csv(args.out, index=False)
 
    print(f"\n[BASARILI] {args.out}")
    print(f"Boyut: {final.shape}  "
          f"(1 ISO + {reduced.shape[1]} PCA + {len(TARGET_FEATURES)} hedef)")
    print(f"Dil sayisi: {len(final)}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 