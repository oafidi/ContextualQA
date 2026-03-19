import json

INPUT_FILE = "export.json"          # fichier exporté depuis Label Studio
OUTPUT_FILE = "final_dataset.json" # dataset propre final

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

final_data = []

for task in data:
    context = task["data"]["context"]

    # 🔹 valeurs par défaut (ancien dataset)
    shhal = task["data"].get("has_shhal")
    shno  = task["data"].get("has_shno")

    corrected = 0  # flag pour savoir si modifié

    # 🔹 si annotation existe → override
    if "annotations" in task and len(task["annotations"]) > 0:
        ann = task["annotations"][-1]  # prend la dernière annotation

        if len(ann.get("result", [])) > 0:
            for res in ann["result"]:
                if res["from_name"] == "shhal":
                    shhal = int(res["value"]["choices"][0])
                    corrected = 1
                elif res["from_name"] == "shno":
                    shno = int(res["value"]["choices"][0])
                    corrected = 1

    final_data.append({
        "context": context,
        "has_shhal": shhal,
        "has_shno": shno,
        "corrected": corrected  # 🔥 utile pour analyse
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print(f"✅ Final dataset saved: {len(final_data)} samples")