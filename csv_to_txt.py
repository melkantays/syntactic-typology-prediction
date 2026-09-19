"""csv_to_txt.py
 
melkan_elamite_cleaned.csv'nin transliteration sutununu alip
temiz, virgulsuz, baslikliksiz bir duz metne cevirir.
Cikti: UD_Elx/elx.txt  (feature_engineer.py'nin bekledigi format)
"""
 
import csv
import os
 
INPUT = "melkan_elamite_cleaned.csv"
OUT_DIR = "UD_Elx"
OUT_FILE = os.path.join(OUT_DIR, "elx.txt")
 
os.makedirs(OUT_DIR, exist_ok=True)
 
words = []
with open(INPUT, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)          # baslik satirini otomatik atlar
    for row in reader:
        text = row.get("transliteration", "").strip()
        if text:                        # bos satirlari alma
            words.append(text)
 
# Tum parcalari boslukla birlestir: tek uzun metin havuzu
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(" ".join(words))
 
print(f"[TAMAM] {len(words)} parca -> {OUT_FILE}")
print(f"Toplam kelime (kabaca): {len(' '.join(words).split())}")
 
