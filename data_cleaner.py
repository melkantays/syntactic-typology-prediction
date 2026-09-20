"""data_cleaner.py
 
Strips editorial and philological apparatus from the raw ORACC
transliterations, leaving plain word forms.
 
Input  : melkan_elamite_dataset.csv   (raw ORACC extraction)
Output : melkan_elamite_cleaned.csv   (transliteration + translation)
 
What is removed, and why:
  {...}        determinatives (semantic classifiers written but not read)
  [ ] ( )      editorial brackets marking damaged or restored signs; the
               bracket is removed, the sign inside is kept
  -            sign separators within a word: hu-ut-tas -> huttas, which
               is what a word-level statistic should see
  subscripts   index numerals disambiguating homophonous signs (ta, ta2,
               ta3, ...). These are an editorial convention, not part of
               the word, so they are stripped.
  x, ...       placeholders for illegible or missing signs: not words,
               and counting them inflates the type count.
 
FIXED: the subscript class previously listed only U+2081-U+2084
(subscripts 1-4). Subscripts 5-9 occur in Elamite transliteration
(ka5, li9, sa6) and survived cleaning, so the same word form could be
counted as two distinct TYPES -- inflating the type-token ratio, which
is one of the three surface features feeding the prediction. The class
now covers the full U+2080-U+2089 range plus ASCII digits.
"""
 
import csv
import re
 
input_file = "melkan_elamite_dataset.csv"
output_file = "melkan_elamite_cleaned.csv"
 
# Full subscript range U+2080-U+2089, plus ASCII digits and the hyphen.
APPARATUS_RE = re.compile(r"[\-0-9₀-₉]")
DETERMINATIVE_RE = re.compile(r"\{.*?\}")
BRACKET_RE = re.compile(r"[\[\]\(\)]")
# A token that is only a break marker carries no lexical content.
BREAK_RE = re.compile(r"^(?:x+|\.+|o+)$", re.IGNORECASE)
 
 
def clean_cuneiform(text):
    """Remove the editorial apparatus and return the bare word form."""
    text = DETERMINATIVE_RE.sub("", text)
    text = BRACKET_RE.sub("", text)
    text = APPARATUS_RE.sub("", text)
    return text.strip()
 
 
cleaned_data = []
dropped_breaks = 0
 
print("Initiating data purge...")
 
with open(input_file, "r", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        original = row.get("transliteration", "")
        translation = row.get("translation", "")
 
        cleaned_word = clean_cuneiform(original)
 
        if not cleaned_word:
            continue
        if BREAK_RE.match(cleaned_word):
            # Illegible / broken sign placeholder, not a word form.
            dropped_breaks += 1
            continue
 
        cleaned_data.append({
            "transliteration": cleaned_word,
            "translation": translation,
        })
 
with open(output_file, "w", encoding="utf-8", newline="") as outfile:
    writer = csv.DictWriter(
        outfile, fieldnames=["transliteration", "translation"]
    )
    writer.writeheader()
    writer.writerows(cleaned_data)
 
print(f"Purge complete. {len(cleaned_data)} Elamite word forms "
      f"saved to {output_file}.")
if dropped_breaks:
    print(f"Dropped {dropped_breaks} break/illegible-sign placeholder(s).")
 