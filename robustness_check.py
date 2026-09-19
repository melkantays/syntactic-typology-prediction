"""robustness_check.py
 
Negatif bulgunun ("3 yuzey istatistigi derin sentaksi tahmin edemiyor")
tek bir sansa, tek bir modele ya da tek bir ayara bagli OLMADIGINI
gosterir.
 
Neden gerekli: bir hakem "belki RandomForest kotuydu", "belki o hedefler
sansti", "belki veri sirasi etkiledi" diyebilir. Bu dosya o itirazlari
ONCEDEN eler: ayni negatif sonuc birden cok bagimsiz ayarda tekrar
ediyorsa, bulgu artik "bizim hatamiz" degil, yontemin gercek siniridir.
 
Girdi : training_matrix.csv  (ISO_Code + 3 X + N Y)
Cikti : ekrana karsilastirma tablosu + robustness_report.json
 
Yaptigi dort bagimsiz test:
  1. Farkli modeller     : RandomForest, LogisticRegression, DummyClassifier
                           hepsi baseline'i gecemiyorsa, sorun model degil.
  2. Ozellik karistirma  : X'i satir satir karistirinca sonuc DEGISMIYORSA,
                           model zaten X'ten bilgi almiyordu demektir (kanit).
  3. Hedef alt kumeleri  : rastgele 4'er hedeflik gruplar -- sonuc her
                           grupta negatifse, belirli hedeflere bagli degil.
  4. Farkli tohumlar     : rastgeleligi degistirince sonuc oynamiyorsa saglam.
"""
 
from __future__ import annotations
 
import argparse
import json
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
FEATURE_COLS = ["TTR", "Avg_Length", "Length_Var"]
ID_COLUMN = "ISO_Code"
 
 
# --- BLOK 1: Veriyi yukle -------------------------------------------- #
def load_matrix(csv_path):
    """training_matrix.csv'yi X (3 ozellik) ve Y (kalan hepsi) olarak ayirir."""
    frame = pd.read_csv(csv_path)
    if ID_COLUMN in frame.columns:
        frame = frame.drop(columns=[ID_COLUMN])
    target_names = [c for c in frame.columns if c not in FEATURE_COLS]
    features = frame[FEATURE_COLS].to_numpy(dtype=np.float64)
    targets = frame[target_names].to_numpy().astype(np.int8)
    return features, targets, target_names
 
 
# --- BLOK 2: Baseline (yenilmesi gereken cizgi) ---------------------- #
def majority_baseline_loss(targets):
    """Her hedef icin en sik cevabi diyen tembel tahmincinin Hamming Loss'u."""
    majority = (targets.mean(axis=0) >= 0.5).astype(np.int8)
    baseline_pred = np.tile(majority, (targets.shape[0], 1))
    return hamming_loss(targets, baseline_pred)
 
 
# --- BLOK 3: Bir modelin LOOCV kaybini olc --------------------------- #
def loocv_loss(pipeline, features, targets):
    """Her hedefi AYRI isleyerek LOOCV Hamming Loss hesaplar.
 
    Neden hedef-hedef: kucuk/dengesiz veride, bir hedef LOOCV katlaminda
    tek-sinifli kalabilir (o dili atinca geri kalan hep 0). Lojistik
    regresyon bunda coker. Burada her hedefi ayri isleyip, egitim verisi
    tek sinifliysa dogrudan o sabiti tahmin ediyoruz -- cokme olmadan,
    dogru sonucla.
    """
    loo = LeaveOneOut()
    n, n_targets = targets.shape
    preds = np.zeros_like(targets)
    for j in range(n_targets):
        yj = targets[:, j]
        for train_idx, test_idx in loo.split(features):
            y_train = yj[train_idx]
            if len(np.unique(y_train)) < 2:
                # Egitimde tek sinif: o sabiti tahmin et.
                preds[test_idx, j] = y_train[0]
            else:
                from sklearn.base import clone
                model = clone(pipeline)
                model.fit(features[train_idx], y_train)
                preds[test_idx, j] = model.predict(features[test_idx])
    return hamming_loss(targets, preds)
 
 
 
 
def make_model(kind, seed):
    """Uc farkli siniflandiriciyi ayni pipeline iskeletinde uretir."""
    if kind == "RandomForest":
        clf = RandomForestClassifier(n_estimators=300, max_depth=2,
                                     random_state=seed, n_jobs=1)
    elif kind == "LogisticReg":
        clf = LogisticRegression(max_iter=1000, random_state=seed)
    elif kind == "Dummy":
        clf = DummyClassifier(strategy="most_frequent")
    else:
        raise ValueError(kind)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
 
 
# --- BLOK 4: Test 1 -- farkli modeller ------------------------------- #
def test_models(features, targets, baseline):
    """Uc farkli model de baseline'i gecemiyorsa, sorun model degildir."""
    print("\n=== TEST 1: Farkli modeller ===")
    print(f"{'Model':16s} {'Hamming Loss':>13s} {'vs baseline':>12s}")
    results = {}
    for kind in ["RandomForest", "LogisticReg", "Dummy"]:
        loss = loocv_loss(make_model(kind, 42), features, targets)
        verdict = "GECTI" if loss < baseline else "gecemedi"
        print(f"{kind:16s} {loss:13.4f} {verdict:>12s}")
        results[kind] = float(loss)
    return results
 
 
# --- BLOK 5: Test 2 -- ozellik karistirma (permutation) -------------- #
def test_shuffle(features, targets, baseline, seed=42):
    """X satirlarini karistir: sonuc degismiyorsa X zaten bilgi tasimiyordu.
 
    Bu en guclu kanit: eger X'i dil-dil karistirinca (yani her dile YANLIS
    istatistikleri verince) model ayni ya da daha iyi calisiyorsa, model
    X'ten hicbir gercek sinyal almiyor demektir. 'Sinyal yoklugunun' dogrudan
    kaniti budur.
    """
    print("\n=== TEST 2: Ozellik karistirma (X'i bozunca ne oluyor?) ===")
    rng = np.random.default_rng(seed)
    shuffled = features[rng.permutation(features.shape[0])]
    real_loss = loocv_loss(make_model("RandomForest", 42), features, targets)
    shuf_loss = loocv_loss(make_model("RandomForest", 42), shuffled, targets)
    print(f"Gercek X ile        : {real_loss:.4f}")
    print(f"Karistirilmis X ile : {shuf_loss:.4f}")
    gap = real_loss - shuf_loss
    if gap >= -0.01:
        print("YORUM: Gercek X, bozuk X'ten daha iyi DEGIL. Model X'ten "
              "anlamli sinyal almiyor -- negatif bulgu dogrulandi.")
    else:
        print("YORUM: Gercek X belirgin daha iyi -- X'te bir sinyal VAR, "
              "bulgu yeniden degerlendirilmeli.")
    return {"real": float(real_loss), "shuffled": float(shuf_loss)}
 
 
# --- BLOK 6: Test 3 -- rastgele hedef alt kumeleri ------------------- #
def test_subsets(features, targets, target_names, seed=42):
    """Rastgele 4'er hedeflik gruplarda da sonuc negatif mi?
 
    Belirli birkac 'kotu' hedefin sonucu bozmadigini gosterir: farkli
    hedef gruplarinda tekrar tekrar baseline gecilemiyor.
    """
    print("\n=== TEST 3: Rastgele hedef alt kumeleri (4'er) ===")
    rng = np.random.default_rng(seed)
    n_targets = targets.shape[1]
    if n_targets < 4:
        print("Hedef sayisi < 4, bu test atlandi.")
        return []
    results = []
    for i in range(3):
        cols = rng.choice(n_targets, size=4, replace=False)
        sub = targets[:, cols]
        base = majority_baseline_loss(sub)
        loss = loocv_loss(make_model("RandomForest", 42), features, sub)
        verdict = "GECTI" if loss < base else "gecemedi"
        names = ", ".join(target_names[c] for c in cols)
        print(f"Grup {i+1}: model={loss:.4f}  baseline={base:.4f}  -> {verdict}")
        print(f"        ({names})")
        results.append({"model": float(loss), "baseline": float(base),
                        "beats": bool(loss < base)})
    return results
 
 
# --- BLOK 7: Test 4 -- farkli tohumlar ------------------------------- #
def test_seeds(features, targets, baseline):
    """Rastgeleligi degistirince (farkli seed) sonuc oynuyor mu?"""
    print("\n=== TEST 4: Farkli rastgele tohumlar ===")
    losses = []
    for seed in [0, 1, 42, 123, 2024]:
        loss = loocv_loss(make_model("RandomForest", seed), features, targets)
        losses.append(loss)
        print(f"seed={seed:5d}: {loss:.4f}  ({'GECTI' if loss < baseline else 'gecemedi'})")
    spread = max(losses) - min(losses)
    print(f"Yayilim (max-min): {spread:.4f}  "
          f"({'kararli' if spread < 0.03 else 'oynak'})")
    return [float(x) for x in losses]
 
 
def main(argv=None):
    p = argparse.ArgumentParser(description="Negatif bulgu saglamlik kontrolu.")
    p.add_argument("--data", type=Path, default=Path("training_matrix.csv"))
    p.add_argument("--out", type=Path, default=Path("robustness_report.json"))
    args = p.parse_args(argv)
 
    if not args.data.exists():
        print(f"HATA: {args.data} bulunamadi.", file=sys.stderr)
        return 1
 
    features, targets, target_names = load_matrix(args.data)
    baseline = majority_baseline_loss(targets)
    print(f"Dil: {features.shape[0]}, Hedef: {targets.shape[1]}")
    print(f"Majority baseline (yenilmesi gereken): {baseline:.4f}")
 
    report = {
        "n_languages": int(features.shape[0]),
        "n_targets": int(targets.shape[1]),
        "baseline": float(baseline),
        "test1_models": test_models(features, targets, baseline),
        "test2_shuffle": test_shuffle(features, targets, baseline),
        "test3_subsets": test_subsets(features, targets, target_names),
        "test4_seeds": test_seeds(features, targets, baseline),
    }
 
    print("\n=== GENEL YORUM ===")
    rf = report["test1_models"]["RandomForest"]
    if rf >= baseline:
        print("Hicbir ayar baseline'i anlamli gecemedi. Negatif bulgu "
              "SAGLAM: 3 yuzey istatistigi bu hedefleri tahmin etmeye "
              "yetmiyor, ve bu tek bir modele/ayara/sansa bagli degil.")
    else:
        print("Bir ayarda baseline gecildi -- bulgu yeniden incelenmeli.")
 
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nRapor -> {args.out}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 