import os
import subprocess
import sys

# 1. Force install Wikipedia API wrapper
subprocess.check_call([sys.executable, "-m", "pip", "install", "wikipedia"])
import wikipedia

# 2. ISO Mapping (lang2vec 3-letter -> Wikipedia 2-letter)
lang_map = {
    "tur": "tr", "eng": "en", "ara": "ar", "fin": "fi", 
    "hun": "hu", "eus": "eu", "fas": "fa", "hin": "hi", 
    "jpn": "ja", "kor": "ko", "heb": "he", "ind": "id", 
    "vie": "vi", "tam": "ta", "rus": "ru", "deu": "de", 
    "fra": "fr", "ita": "it", "spa": "es", "cmn": "zh"
}

# 3. Universal search targets (The word "History" in 20 languages)
search_terms = {
    "tr": "Tarih", "en": "History", "ar": "تاريخ", "fi": "Historia",
    "hu": "Történelem", "eu": "Historia", "fa": "تاریخ", "hi": "इतिहास",
    "ja": "歴史", "ko": "역사", "he": "היסטוריה", "id": "Sejarah",
    "vi": "Lịch sử", "ta": "வரலாறு", "ru": "История", "de": "Geschichte",
    "fr": "Histoire", "it": "Storia", "es": "Historia", "zh": "历史"
}

target_folder = "UD_Corpus"
os.makedirs(target_folder, exist_ok=True)
print("\n--- INITIATING WIKIPEDIA DATA MINING (20 LANGUAGES) ---")

for iso_3, iso_2 in lang_map.items():
    file_path = f"{target_folder}/{iso_3}.txt"
    
    if os.path.exists(file_path):
        print(f"[{iso_3.upper()}] Data already exists.")
        continue
        
    try:
        print(f"[{iso_3.upper()}] Mining Wikipedia ({iso_2})...")
        wikipedia.set_lang(iso_2)
        
        # Pull raw text from Wikipedia
        page = wikipedia.page(search_terms[iso_2], auto_suggest=False)
        text_data = page.content
        
        # Seal into the project folder
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_data)
    except Exception as e:
        print(f" -> ERROR ({iso_3}): Blocked. Moving to next.")

print(f"\n[OPERATION COMPLETED] Raw text data sealed in '{target_folder}'.")