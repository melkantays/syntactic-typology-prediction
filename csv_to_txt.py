"""csv_to_txt.py
 
Takes the `transliteration` column of melkan_elamite_cleaned.csv and
writes it out as clean plain text -- no commas, no header row.
Output: UD_Elx/elx.txt  (the format feature_engineer.py expects)
 
CAVEAT (must be stated in the paper): the records are joined with spaces
into one long string. The result is a bag of disconnected transliterated
fragments, not running text. Type-token ratio and token length are
word-level statistics and are unaffected. Contextual embeddings are not:
a transformer builds its representation over sentence structure, and no
sentence structure survives this concatenation.
"""
 
import csv
import os
 
INPUT = "melkan_elamite_cleaned.csv"
OUT_DIR = "UD_Elx"
OUT_FILE = os.path.join(OUT_DIR, "elx.txt")
 
os.makedirs(OUT_DIR, exist_ok=True)
 
words = []
with open(INPUT, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)          # skips the header row automatically
    for row in reader:
        text = row.get("transliteration", "").strip()
        if text:                        # drop empty rows
            words.append(text)
 
# Join every fragment with a space: one long pool of text.
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(" ".join(words))
 
print(f"[DONE] {len(words)} fragments -> {OUT_FILE}")
print(f"Approximate word count: {len(' '.join(words).split())}")
 