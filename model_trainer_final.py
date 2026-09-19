"""model_trainer_final.py
 
Nihai modeli egitir ve Elamca icin 7 tipolojik ozelligin olasiligini
uretir. Her satir, bulgunun durumunu tasiyan DAMGALARLA yazilir.
 
EPISTEMIK DURUM (onemli):
Bu modelin LOOCV dogrulamasi BASARISIZ oldu -- uc deneyde de (yuzey,
10-PCA, 20-PCA) majority baseline gecilemedi. Dolayisiyla asagidaki
tahminler DOGRULANMIS dilbilimsel bulgular DEGIL, "model calissaydi ne
derdi" prototipidir. Bu yuzden her cikti satiri, dogrulama sonucunu ve
Elamca'nin URIEL'de sifir gozlenmis verisi oldugunu tasir.
 
Girdi (yuzey modu):
  training_matrix.csv       egitim (31 dil, 3 X + 7 Y)
  elamite_features.csv      Elamca'nin 3 X degeri
Girdi (embedding modu):
  embedding_features.csv    egitim dillerinin 768 boyutu
  elamite_embedding.csv     Elamca'nin 768 boyutu
  training_matrix.csv       hedefler ve dil kumesi icin
 
Cikti: elamite_typology_predictions.csv (uzun format, damgali)
"""
 
from __future__ import annotations
 
import argparse
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
ID_COL = "ISO_Code"
SURFACE_COLS = ["TTR", "Avg_Length", "Length_Var"]
 
 
# --- BLOK 1: Yuzey modu verisi --------------------------------------- #
def load_surface(train_path, query_path):
    """Yuzey deneyinin X'i: 3 istatistik. Elamca da ayni 3 sutundan."""
    train = pd.read_csv(train_path)
    query = pd.read_csv(query_path)
 
    target_names = [c for c in train.columns if c.startswith("S_")]
    x_train = train[SURFACE_COLS].to_numpy(np.float64)
    y_train = train[target_names].to_numpy().astype(np.int8)
    x_query = query[SURFACE_COLS].to_numpy(np.float64)
 
    if len(x_query) != 1:
        raise ValueError(f"Sorgu dosyasinda 1 satir olmali, {len(x_query)} var.")
    return x_train, y_train, x_query, target_names, SURFACE_COLS
 
 
# --- BLOK 2: Embedding modu verisi ----------------------------------- #
def load_embedding(emb_path, query_emb_path, train_path, n_components):
    """Embedding deneyinin X'i: 768 boyut -> PCA.
 
    EN KRITIK NOKTA: PCA ve StandardScaler SADECE egitim dillerinde
    fit edilir; Elamca'ya yalnizca transform uygulanir. Elamca'yi fit'e
    dahil etseydik, tahmin edecegimiz dil donusumu sekillendirmis olurdu
    -- bu bir veri sizintisi turudur ve sonucu gecersiz kilar.
 
    Ayrica egitim dilleri, training_matrix.csv'deki kumeyle kesistirilir;
    boylece onceki deneylerle AYNI 31 dil kullanilir.
    """
    emb = pd.read_csv(emb_path)
    query = pd.read_csv(query_emb_path)
    train_ref = pd.read_csv(train_path)
 
    emb_cols = [c for c in emb.columns if c.startswith("emb_")]
    target_names = [c for c in train_ref.columns if c.startswith("S_")]
 
    # Egitim kumesini referansla kesistir (ayni 31 dil)
    ref_langs = train_ref[ID_COL].tolist()
    emb = emb[emb[ID_COL].isin(ref_langs)].copy()
    emb = emb.set_index(ID_COL).loc[ref_langs].reset_index()
 
    x_raw = emb[emb_cols].to_numpy(np.float64)
    x_query_raw = query[emb_cols].to_numpy(np.float64)
    y_train = train_ref[target_names].to_numpy().astype(np.int8)
 
    # Olcekleme + PCA: SADECE egitimde fit
    scaler = StandardScaler().fit(x_raw)
    pca = PCA(n_components=min(n_components, x_raw.shape[0]),
              random_state=42).fit(scaler.transform(x_raw))
 
    x_train = pca.transform(scaler.transform(x_raw))
    x_query = pca.transform(scaler.transform(x_query_raw))
 
    print(f"PCA: {len(emb_cols)} -> {x_train.shape[1]} boyut "
          f"(korunan varyans {pca.explained_variance_ratio_.sum():.1%})")
    feature_names = [f"pca_{i}" for i in range(x_train.shape[1])]
    return x_train, y_train, x_query, target_names, feature_names
 
 
# --- BLOK 3: Modeli kur ve egit -------------------------------------- #
def fit_model(x_train, y_train, seed=42):
    """Dogrulamada secilen konfigurasyonla nihai modeli egitir.
 
    max_depth=2 uc deneyde de secildi -- model derinlestikce kotulestigi
    icin en sig agacta kaliyor. Bu, ogrenilecek oruntu olmadiginin
    isaretidir; yine de sureklilik icin ayni konfigurasyon kullaniliyor.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500, max_depth=2, min_samples_split=2,
            max_features="sqrt", random_state=seed, n_jobs=-1)),
    ])
    model.fit(x_train, y_train)
    return model
 
 
# --- BLOK 4: Olasiliklari guvenle cikar ------------------------------ #
def positive_probabilities(model, x_query, n_targets):
    """Her hedef icin P(deger=1) degerini dondurur.
 
    predict_proba, cok-hedefli modelde hedef basina AYRI bir dizi doner
    ve her dizinin sutun sayisi o hedefte gorulen SINIF SAYISI kadardir.
    Egitimde sabit kalan bir hedefte (ornegin tum diller 0) tek sutun
    olur; naif [0][1] indeksleme burada IndexError verir. Sinif kimligi
    classes_ uzerinden okunur, indeks varsayilmaz.
    """
    blocks = model.predict_proba(x_query)
    classes = model.named_steps["clf"].classes_
 
    out = np.empty(n_targets, dtype=np.float64)
    for i in range(n_targets):
        cls = np.asarray(classes[i])
        block = np.asarray(blocks[i])[0]
        hit = np.flatnonzero(cls == 1)
        out[i] = float(block[hit[0]]) if hit.size else 0.0
    return out
 
 
# --- BLOK 5: Bilgi degeri teshisi ------------------------------------ #
def diagnose(probs, y_train, eps=0.02):
    """Tahmin, egitim taban oranini mi tekrarliyor?
 
    Bir tahmin taban oranina esitse, model o hedefte Elamca'ya OZGU
    hicbir sey sOylememis, sadece korpus ortalamasini yansitmistir.
    Bunlari 'VACUOUS' diye isaretliyoruz -- hocaya "103'un su kadari
    bilgi tasimiyor" diyebilmek icin.
    """
    base = y_train.mean(axis=0)
    delta = np.abs(probs - base)
    degenerate = (base == 0.0) | (base == 1.0)
    vacuous = (delta < eps) & ~degenerate
    return base, delta, degenerate, vacuous
 
 
# --- BLOK 6: Damgali ciktiyi yaz ------------------------------------- #
def write_output(path, mode, target_names, probs, base, delta,
                 degenerate, vacuous, n_train, n_tokens):
    """Uzun formatta, her satiri damgali CSV yazar.
 
    Uzun format bilincli: 1x7'lik genis satirda biri tek bir olasiligi
    alip slayta koyar, damga geride kalir. Uzun formatta damga sayiyla
    birlikte seyahat eder.
    """
    status = np.where(degenerate, "DEGENERATE_LABEL",
                      np.where(vacuous, "VACUOUS_EQUALS_PRIOR", "INFORMATIVE"))
    frame = pd.DataFrame({
        "Language": "Elamite",
        "Feature_Set": mode,
        "Typology_Feature": target_names,
        "P_positive": np.round(probs, 6),
        "Training_Base_Rate": np.round(base, 6),
        "Delta_From_Prior": np.round(delta, 6),
        "Status": status,
        "Model_Validation": "FAILED_LOOCV_NOT_ABOVE_BASELINE",
        "URIEL_Observed_Data_For_Elamite": 0,
        "Training_Languages": n_train,
        "Elamite_Tokens": n_tokens,
    })
    frame.to_csv(path, index=False)
    return frame
 
 
def main(argv=None):
    p = argparse.ArgumentParser(description="Elamca tipoloji prototipi.")
    p.add_argument("--mode", choices=["surface", "embedding"], default="surface")
    p.add_argument("--train", type=Path, default=Path("training_matrix.csv"))
    p.add_argument("--query", type=Path, default=Path("elamite_features.csv"))
    p.add_argument("--emb", type=Path, default=Path("embedding_features.csv"))
    p.add_argument("--query-emb", type=Path, default=Path("elamite_embedding.csv"))
    p.add_argument("--components", type=int, default=10)
    p.add_argument("--out", type=Path,
                   default=Path("elamite_typology_predictions.csv"))
    p.add_argument("--tokens", type=int, default=1362,
                   help="Elamca korpusundaki token sayisi (damga icin).")
    args = p.parse_args(argv)
 
    if args.mode == "surface":
        if not (args.train.exists() and args.query.exists()):
            print("HATA: training_matrix.csv veya elamite_features.csv yok.",
                  file=sys.stderr)
            return 1
        x_tr, y_tr, x_q, targets, feats = load_surface(args.train, args.query)
    else:
        if not (args.emb.exists() and args.query_emb.exists()):
            print("HATA: embedding dosyalari yok.", file=sys.stderr)
            return 1
        x_tr, y_tr, x_q, targets, feats = load_embedding(
            args.emb, args.query_emb, args.train, args.components)
 
    print(f"Mod        : {args.mode}")
    print(f"Egitim     : {x_tr.shape[0]} dil, {x_tr.shape[1]} ozellik")
    print(f"Hedef      : {len(targets)}")
    print("UYARI      : Model dogrulamayi GECEMEDI. Asagidaki tahminler")
    print("             dogrulanmis bulgu DEGIL, prototiptir.\n")
 
    model = fit_model(x_tr, y_tr)
    probs = positive_probabilities(model, x_q, len(targets))
    base, delta, degenerate, vacuous = diagnose(probs, y_tr)
 
    print(f"{'Ozellik':30s} {'P(1)':>7s} {'Taban':>7s} {'Durum':>22s}")
    for i, name in enumerate(targets):
        st = ("DEGENERATE" if degenerate[i]
              else "VACUOUS" if vacuous[i] else "INFORMATIVE")
        print(f"{name:30s} {probs[i]:7.3f} {base[i]:7.3f} {st:>22s}")
 
    n_inf = int((~degenerate & ~vacuous).sum())
    print(f"\nBilgi tasiyan tahmin: {n_inf}/{len(targets)}")
    print(f"Taban orani tekrari : {int(vacuous.sum())}")
    print(f"Dejenere hedef      : {int(degenerate.sum())}")
 
    write_output(args.out, args.mode, targets, probs, base, delta,
                 degenerate, vacuous, x_tr.shape[0], args.tokens)
    print(f"\nCikti -> {args.out}")
    print("Her satir dogrulama durumunu ve Elamca'nin URIEL'de 0 gozlenmis")
    print("verisi oldugunu tasiyor; tek bir sayi baglamdan koparilamaz.")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 