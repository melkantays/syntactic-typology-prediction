import os

# =====================================================================
# ACTION REQUIRED: Update this path to your downloaded UD Treebanks folder
# Example: r"C:\Users\melka\Downloads\ud-treebanks-v2.13"
# =====================================================================
UD_ROOT_PATH = r"C:\Users\melka\Desktop\ud-treebanks-v2.18\ud-treebanks-v2.18"

TARGET_FOLDER = "UD_Corpus"
os.makedirs(TARGET_FOLDER, exist_ok=True)

# Mapping UD 2-letter file prefixes to URIEL 3-letter ISO codes
iso_map = {
    "tr": "tur", "en": "eng", "ar": "ara", "fi": "fin", "hu": "hun",
    "eu": "eus", "fa": "fas", "hi": "hin", "ja": "jpn", "ko": "kor",
    "he": "heb", "id": "ind", "vi": "vie", "ta": "tam", "ru": "rus",
    "de": "deu", "fr": "fra", "it": "ita", "es": "spa", "zh": "cmn",
    "cs": "ces", "pl": "pol", "nl": "nld", "pt": "por", "sv": "swe",
    "bg": "bul", "da": "dan", "el": "ell", "ro": "ron", "uk": "ukr",
    "sk": "slk", "hr": "hrv", "sr": "srp", "sl": "slv", "et": "est",
    "lt": "lit", "lv": "lav", "no": "nob"
}

found_langs = set()

print("\n--- INITIATING LOCAL OFFLINE DATA EXTRACTION ---")

for root, dirs, files in os.walk(UD_ROOT_PATH):
    for file in files:
        # Target the training datasets specifically
        if file.endswith("-ud-train.conllu") or file.endswith("train.conllu"):
            prefix = file.split("_")[0]
            
            if prefix in iso_map:
                iso_3 = iso_map[prefix]
                
                # Prevent overwriting if we already found a corpus for this language
                if iso_3 in found_langs:
                    continue
                    
                file_path = os.path.join(root, file)
                target_file = os.path.join(TARGET_FOLDER, f"{iso_3}.txt")
                
                print(f"[{iso_3.upper()}] Extracting raw text from: {file}")
                
                with open(file_path, "r", encoding="utf-8") as infile, \
                     open(target_file, "w", encoding="utf-8") as outfile:
                    
                    for line in infile:
                        # UD conllu files store the raw sentence after '# text = '
                        if line.startswith("# text = "):
                            clean_text = line.replace("# text = ", "").strip()
                            outfile.write(clean_text + "\n")
                            
                found_langs.add(iso_3)

print(f"\n[OPERATION COMPLETED] Successfully extracted {len(found_langs)} languages into '{TARGET_FOLDER}'.")