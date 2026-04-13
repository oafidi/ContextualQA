import pandas as pd
import json
import time, os, sys
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

INPUT_FILE = "./validation/part2.csv"
OUTPUT_FILE = "generated_dataset.json"
MODEL = "gpt-4o"
START_INDEX = 0
END_INDEX = 5

# ================= LOAD DATA =================
df = pd.read_csv(INPUT_FILE)
df_slice = df.iloc[START_INDEX:END_INDEX]
contexts = df_slice["context"].dropna().tolist()

# ================= PROMPT =================
PROMPT_TEMPLATE = """نتا خبير ف الداريجة المغربية. مهمتك هي تكتب سؤال واحد بالداريجة والجواب ديالو مباشرة من النص.

القواعد:
- السؤال خاصو يبدا بـ: شكون / فين / فوقاش / معامن / وقتاش / شنو / شحال
- السؤال خاصو ينتهي بـ ؟
- الجواب خاصو يكون مقتطف حرفي من النص
- الرد JSON فقط بالشكل:
{{"question": "...", "answer": "..."}}

النص:
{context}

الرد:
"""

# ================= MODEL CALL =================
def generate_qa(context: str):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON with standard ASCII quotes."},
            {"role": "user", "content": PROMPT_TEMPLATE.format(context=context)}
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content
    # Fix smart/curly quotes -> standard quotes
    raw = raw.replace('\u201c', '"').replace('\u201d', '"')
    raw = raw.replace('\u2018', "'").replace('\u2019', "'")
    return json.loads(raw)

# ================= LOAD CHECKPOINT =================
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} existing tasks")
else:
    tasks = []

processed_texts = set(task["data"]["context"] for task in tasks)

# ================= MAIN LOOP =================
for context in tqdm(contexts):
    if context in processed_texts:
        continue

    try:
        result = generate_qa(context)

        tasks.append({
            "data": {
                "context": context,
                "question": result["question"],
                "answer": result["answer"]
            }
        })

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

        time.sleep(0.3)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        continue

print(f"\n✅ Done — Saved to {OUTPUT_FILE}")