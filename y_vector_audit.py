import lang2vec.lang2vec as l2v

iso_mapping = {
    "tur": "Turkish", "akk": "Akkadian", "amh": "Amharic",
    "hbo": "Ancient Hebrew", "grc": "Ancient Greek", "ara": "Arabic",
    "hye": "Armenian", "eus": "Basque", "aii": "Assyrian",
    "cpg": "Cappadocian", "ckb": "Central Kurdish", "cop": "Coptic",
    "eng": "English", "jpn": "Japanese", "hit": "Hittite"
}

print("Initiating URIEL Database Audit for Y-Vectors...\n")

valid_langs = []
invalid_langs = []

for code, name in iso_mapping.items():
    try:
        feat = l2v.get_features([code], "syntax_knn")
        if code in feat:
            valid_langs.append(name)
        else:
            invalid_langs.append(name)
    except:
        invalid_langs.append(name)

print(f"✅ SUPPORTED LANGUAGES (Ready for Training):")
for v in valid_langs:
    print(f" - {v}")

print(f"\n❌ MISSING LANGUAGES (Must be Discarded):")
for i in invalid_langs:
    print(f" - {i}")