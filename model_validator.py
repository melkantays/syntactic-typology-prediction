"""model_validator.py
 
training_matrix.csv'yi alir, Random Forest modelini LOOCV (Leave-One-Out
Cross-Validation) ile test eder ve sonucu APTAL BIR TAHMINCIYLE (majority
baseline) karsilastirir.
 
Neden bu dosya var: bir modelin "iyi" olup olmadigini, dogruluk (accuracy)
gibi kolay kandiran bir sayiya bakarak degil, "her seye en sik cevabi
diyen" aptal bir tahminciyi yenip yenemedigine bakarak anlariz. Yenemiyorsa
model degersizdir -- ne kadar sik gorunurse gorunsun.
 
Girdi  : training_matrix.csv  (ISO_Code + 3 X + N tane Y)
Cikti  : ekrana rapor + optimised_pipeline.joblib + validation_report.json
 
NOT: Hedef sayisi ARTIK SABIT DEGIL. Dosyada kac Y sutunu varsa onu alir
(7 de olsa, 6 da olsa). Boylece hedef sayisini degistirince bu dosyaya
dokunmana gerek kalmaz.
"""
 
from __future__ import annotations
 
import argparse
import json
import sys
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, make_scorer
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
RANDOM_STATE = 42
ID_COLUMN = "ISO_Code"
# X ozelliklerinin isimleri sabit; geri kalan her sutun bir Y hedefidir.
FEATURE_COLS = ["TTR", "Avg_Length", "Length_Var"]
 
# Denenecek hiperparametre kombinasyonlari. Kucuk veride modelin
# KAPASITESI degil, ASIRI OGRENMEYE (overfitting) karsi FRENLERI onemli
# oldugu icin, derinlik (max_depth) ve bolunme (min_samples_split) gibi
# frenleri tariyoruz.
PARAM_GRID = {
    "clf__n_estimators": [100, 300, 500],
    "clf__max_depth": [2, 3, 5, None],
    "clf__min_samples_split": [2, 3, 4],
    "clf__max_features": ["sqrt", None],
}
 
 
# --- BLOK 1: Veriyi oku, X ve Y'yi ayir ------------------------------ #
def load_matrix(csv_path):
    """Matrisi X (ozellikler) ve Y (hedefler) olarak ayirir.
 
    ISO_Code sutunu ATILIR -- ozellik olarak kullanilmaz, cunku her dile
    ozgu benzersiz bir etiket; model onu "ezber anahtari" gibi kullanip
    sahte basari uretebilir. Y sutun sayisi dosyadan OKUNUR, sabit degil.
    """
    frame = pd.read_csv(csv_path)
    if ID_COLUMN in frame.columns:
        frame = frame.drop(columns=[ID_COLUMN])
 
    # X ve Y sutunlari SABIT DEGIL, dosyadan tespit edilir:
    #   Y hedefleri "S_" ile baslar (URIEL sentaks ozellikleri).
    #   Geri kalan her sutun bir X ozelligidir.
    # Boylece ayni kod hem yuzey matrisini (TTR, Avg_Length, Length_Var)
    # hem embedding matrisini (pca_0..pca_9) isler -- iki deneyi ayni
    # kodla kosmak karsilastirmayi adil kilar.
    target_names = [c for c in frame.columns if c.startswith("S_")]
    if not target_names:
        raise ValueError("Hic 'S_' ile baslayan Y hedef sutunu yok.")
 
    feature_names = [c for c in frame.columns if c not in target_names]
    if not feature_names:
        raise ValueError("Hic X ozellik sutunu bulunamadi.")
 
    features = frame[feature_names].to_numpy(dtype=np.float64)
    targets = frame[target_names].to_numpy()
 
    # Guvenlik: X'te bosluk olmamali, Y ikili (0/1) olmali.
    if not np.isfinite(features).all():
        raise ValueError("X ozelliklerinde NaN/inf var; once temizle.")
    observed = np.unique(targets)
    if not np.isin(observed, [0, 1]).all():
        raise ValueError(f"Y hedefleri 0/1 olmali; bulunan: {observed[:8]}")
 
    return features, targets.astype(np.int8), feature_names, target_names
 
 
# --- BLOK 2: Aptal tahminci (baseline) hesapla ----------------------- #
def majority_baseline(targets, target_names):
    """Her hedef icin "en sik gorulen cevabi" korukorune diyen tahminci.
 
    Bu, yenilmesi gereken CIZGIDIR. Modelimiz bunun Hamming Loss'unu
    gecemezse, "zeka" katmamis demektir. Ayrica DEJENERE hedefleri
    (tum dillerde ayni deger) sayar -- onlar sahte kolay puandir.
    """
    n = targets.shape[0]
    positive_rate = targets.mean(axis=0)
    constant = (positive_rate == 0.0) | (positive_rate == 1.0)
 
    print("--- Label geometry ---")
    print(f"Samples (n)        : {n}")
    print(f"Targets               : {targets.shape[1]}")
    print(f"Degenerate labels: {int(constant.sum())}")
    print(f"Informative labels   : {int((~constant).sum())}")
 
    majority = (positive_rate >= 0.5).astype(np.int8)
    baseline_pred = np.tile(majority, (n, 1))
    baseline_loss = hamming_loss(targets, baseline_pred)
    print(f"Majority baseline     : {baseline_loss:.4f} (Hamming Loss)\n")
    return baseline_loss
 
 
# --- BLOK 3: Model iskeletini kur ------------------------------------ #
def build_pipeline():
    """Olcekleme + Random Forest'i tek bir boru hatti (pipeline) yapar.
 
    Her sey pipeline icinde oldugu icin, olcekleme her katlamada
    (fold) SADECE egitim verisinden ogrenilir -- test dili sizmaz.
    Buna veri sizintisini onleme denir.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
    ])
 
 
# --- BLOK 4: En iyi hiperparametreleri LOOCV ile ara ----------------- #
def run_search(features, targets, n_jobs):
    """GridSearchCV, her kombinasyonu LOOCV ile deneyip en iyisini secer.
 
    LOOCV: 31 dil varsa, 30'uyla egit + 1'inde test et, bunu 31 kez
    tekrarla. Kucuk veride en dogru test yontemi budur.
 
    Hamming Loss bir HATA olcusu (kucuk=iyi), ama GridSearch puani
    BUYUTMEYE calisir. make_scorer(greater_is_better=False) skoru
    negatiflestirir; boylece "en az hata" = "en buyuk negatif skor".
    """
    scorer = make_scorer(hamming_loss, greater_is_better=False)
    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=PARAM_GRID,
        scoring=scorer,
        cv=LeaveOneOut(),
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(features, targets)
    return search
 
 
# --- BLOK 5: Secilen konfigin DURUST kaybini olc --------------------- #
def honest_loss(estimator, features, targets):
    """Secilen model icin gercek LOOCV kaybi.
 
    Dikkat: GridSearch'in "en iyi skoru", denenen ONLARCA konfig
    icinden en dususudur -- yani secim yanliligiyla iyimserdir.
    Bu fonksiyon, TEK ve SABIT en iyi konfigi bagimsizca yeniden
    olcer; savunulmasi gereken sayi budur.
    """
    predictions = cross_val_predict(
        estimator, features, targets, cv=LeaveOneOut(), n_jobs=1
    )
    return hamming_loss(targets, predictions)
 
 
# --- BLOK 6: Raporla ve kaydet --------------------------------------- #
def report(search, features, targets, baseline_loss, feat_names,
           target_names, model_path, report_path):
    """Sonuclari ekrana basar, modeli ve JSON ozeti diske kaydeder."""
    grid_loss = -search.best_score_
 
    print("--- Best configuration ---")
    for key in sorted(search.best_params_):
        print(f"{key:26s}: {search.best_params_[key]}")
 
    print("\n--- Performance ---")
    print(f"Model Hamming Loss (grid)    : {grid_loss:.4f}")
    print(f"Majority baseline (Hamming Loss)  : {baseline_loss:.4f}")
 
    fixed_loss = honest_loss(search.best_estimator_, features, targets)
    print(f"Model Hamming Loss (honest LOOCV)  : {fixed_loss:.4f}")
 
    lift = baseline_loss - fixed_loss
    print(f"\nImprovement           : {lift:+.4f}")
    if lift <= 0:
        print("SONUC: Model aptal tahminciyi GECEMEDI. Pozitif sonuc "
              "olarak sunma. Bu, yuzey istatistiklerinin yetersizliginin\n"
              "       kaniti -- durust bir negatif bulgu.")
    else:
        print(f"SONUC: Model aptal tahminciyi GECTI ({lift/baseline_loss:.1%} "
              "daha az hata). Savunulabilir pozitif sonuc.")
 
    forest = search.best_estimator_.named_steps["clf"]
    print("\n--- Feature importance ---")
    for name, w in zip(feat_names, forest.feature_importances_):
        print(f"{name:14s}: {w:.4f}")
 
    joblib.dump(search.best_estimator_, model_path)
    summary = {
        "n_languages": int(features.shape[0]),
        "n_targets": len(target_names),
        "target_names": target_names,
        "grid_loss": float(grid_loss),
        "honest_loss": float(fixed_loss),
        "baseline_loss": float(baseline_loss),
        "beats_baseline": bool(lift > 0),
        "best_params": search.best_params_,
    }
    Path(report_path).write_text(json.dumps(summary, indent=2))
    print(f"\nModel  -> {model_path}")
    print(f"Rapor  -> {report_path}")
 
 
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LOOCV validator (esnek hedef sayisi).")
    p.add_argument("--data", type=Path, default=Path("training_matrix.csv"))
    p.add_argument("--model-out", type=Path, default=Path("optimised_pipeline.joblib"))
    p.add_argument("--report-out", type=Path, default=Path("validation_report.json"))
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args(argv)
 
 
def main(argv=None):
    args = parse_args(argv)
    if not args.data.exists():
        print(f"HATA: {args.data} bulunamadi.", file=sys.stderr)
        return 1
 
    features, targets, feat_names, target_names = load_matrix(args.data)
    baseline_loss = majority_baseline(targets, target_names)
 
    print(f"LOOCV araniyor ({features.shape[0]} dil, "
          f"{len(target_names)} hedef)...\n")
    search = run_search(features, targets, args.n_jobs)
    report(search, features, targets, baseline_loss, feat_names,
           target_names, args.model_out, args.report_out)
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 