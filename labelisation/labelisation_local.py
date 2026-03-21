from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json, os
import pandas as pd
from tqdm import tqdm

START_INDEX = 0
END_INDEX   = 1525
OUTPUT_JSON = "datasets/processed/question_generated_by_llm/questions_1_1525.json"
INPUT_PATH  = "datasets/processed/goud_preprocessed_V2.csv"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# model_name = "Qwen/Qwen2-1.5B-Instruct"
model_name = "MBZUAI-Paris/Atlas-Chat-9B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)

def generate_q(context, questions):
    prompt = f"""" نتا خبير ف الداريجة المغربية .

قرا هاد النص مزيان:
\"\"\"
{context}
\"\"\"

عطيني سؤال واحد بالداريجة:
- {questions} يبدا بـ:
- ممنوع: علاش / كيفاش
- الجواب ديالو  خاص يكون موجود صريح ف النص
- جملة واحدة تنتهي بـ؟
- كتب غير السؤال، بلا أي حاجة أخرى

السؤال:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice cleanly after the prompt tokens
    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Keep only the first line (first question)
    question = response.split("\n")[0].strip()

    return question


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, encoding="utf-8")
    print(f"{len(df)} articles loaded")

    df_slice = df.iloc[START_INDEX:END_INDEX]
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        print(f"Loaded {len(tasks)} existing tasks")
    else:
        tasks = []

    processed_texts = set(task["data"]["context"] for task in tasks)
    
    question_types = {
            "has_shno" : "/ آش / شنو /",
            "has_shhal": "/ شحال /",
            "has_location": "/ فين /",
            "has_date": "/ فوقاش / امتا / وقتاش / معاش /",
            "has_person": "/ شكون / معامن /"
        }
    columns = ["has_shno", "has_shhal", "has_location", "has_date", "has_person"]

    for index, row in tqdm(df_slice.iterrows(), total=len(df_slice)):

        context = row["context"]

        if context in processed_texts:
            continue

        questions = ""
        for column in columns:
            if int(row[column]) == 1:
                questions += question_types[column]
        
        questions = questions.replace("//", "/")

        entry = {
            "data" : {
                "context": context,
                "question": generate_q(context, questions)
            }
        }
        tasks.append(entry)

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)