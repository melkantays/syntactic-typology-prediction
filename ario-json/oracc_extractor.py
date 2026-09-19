import json
import os
import csv
def extract_words(node, target_lang="elx"):
    results = []
    
    if isinstance(node, dict):
        if node.get("node") == "l" and "f" in node:
            lang = node["f"].get("lang")
            
            if lang == target_lang:
                form = node["f"].get("form", "")
                gw = node["f"].get("gw", "")
                results.append({"transliteration": form, "translation": gw})
                
        for key, value in node.items():
            results.extend(extract_words(value, target_lang))
            
    elif isinstance(node, list):
        for item in node:
            results.extend(extract_words(item, target_lang))
            
    return results

all_elamite_data = []
folder_path = r"C:\Users\melka\OneDrive\Bilder\Masaüstü\ario-json\ario-json\corpusjson"
print("Initiating bulk extraction across all JSON files...")

for filename in os.listdir(folder_path):
    if filename.endswith(".json") and not filename.startswith("._"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                extracted = extract_words(data, target_lang="elx")
                if extracted:
                    all_elamite_data.extend(extracted)
            except json.JSONDecodeError:
                print(f"Error reading {filename}. Skipping.")

print(f"\nExtraction complete. Found {len(all_elamite_data)} Elamite words across the entire corpus.")

print("\n--- DATA SAMPLE ---")
for item in all_elamite_data[:15]:
    print(f"[{item['transliteration']}] -> {item['translation']}")


print(f"\nExtraction complete. Found {len(all_elamite_data)} Elamite words across the entire corpus.")

output_file = "melkan_elamite_dataset.csv"
with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["transliteration", "translation"])
    writer.writeheader()
    writer.writerows(all_elamite_data)

print(f"\nVeri seti {output_file}")