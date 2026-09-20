"""lang_audit.py
 
Counts which LANGUAGES are actually present in the ORACC corpus JSON.
 
Why this exists: the Achaemenid royal inscriptions are trilingual (Old
Persian, Elamite, Akkadian). If the extraction step did not filter by
language, the resulting "Elamite" corpus may contain word forms from
one or both of the other two. That would invalidate everything
downstream, and it is silent -- nothing crashes, the numbers simply
describe the wrong language.
 
ORACC tags every lemma with a `lang` field (elx = Elamite, akk =
Akkadian, peo = Old Persian, qpn = proper noun). This script walks the
JSON recursively and tallies them.
 
Input  : ario-json/corpusjson/*.json
Output : a table on screen. Nothing is written to disk.
"""
 
import json
from collections import Counter
from pathlib import Path
 
CORPUS_DIR = Path("ario-json/corpusjson")
 
LANG_NAMES = {
    "elx": "Elamite",
    "akk": "Akkadian",
    "peo": "Old Persian",
    "qpn": "proper noun",
    "sux": "Sumerian",
}
 
 
def walk(node, counter):
    """Recursively collect every value stored under a 'lang' key."""
    if isinstance(node, dict):
        lang = node.get("lang")
        if isinstance(lang, str):
            counter[lang] += 1
        for value in node.values():
            walk(value, counter)
    elif isinstance(node, list):
        for item in node:
            walk(item, counter)
 
 
def main():
    if not CORPUS_DIR.is_dir():
        print(f"ERROR: {CORPUS_DIR} not found. Run this from the folder "
              f"that contains the ario-json directory.")
        return 1
 
    files = sorted(CORPUS_DIR.glob("*.json"))
    print(f"Scanning {len(files)} corpus file(s)...\n")
 
    total = Counter()
    per_file_langs = Counter()
 
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  unreadable: {path.name} ({exc})")
            continue
        counter = Counter()
        walk(data, counter)
        total.update(counter)
        for lang in counter:
            per_file_langs[lang] += 1
 
    if not total:
        print("No 'lang' field found anywhere. The extraction script must be")
        print("read directly to determine what it pulled.")
        return 0
 
    print(f"{'lang':10s} {'name':14s} {'lemmas':>9s} {'files':>7s} {'share':>8s}")
    grand = sum(total.values())
    for lang, count in total.most_common():
        name = LANG_NAMES.get(lang.split("-")[0], "?")
        print(f"{lang:10s} {name:14s} {count:9d} {per_file_langs[lang]:7d} "
              f"{count/grand:7.1%}")
 
    print(f"\nTotal tagged lemmas: {grand}")
    elx = sum(v for k, v in total.items() if k.startswith("elx"))
    if elx == 0:
        print("\nNO ELAMITE FOUND. Whatever melkan_elamite_dataset.csv holds,")
        print("it is not Elamite. Stop and re-check oracc_extractor.py.")
    else:
        print(f"\nElamite lemmas: {elx} ({elx/grand:.1%} of the corpus).")
        print("If the extractor did not filter on lang == 'elx', the rest of")
        print("that percentage is in your 'Elamite' file too.")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 