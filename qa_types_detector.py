import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import json
import os

START_INDEX = 1525
END_INDEX   = 3050
OUTPUT_JSON = "datasets/processed/question_types_detected_by_llm/qa_types_1526_3050.json "
INPUT_PATH  = "datasets/processed/goud_preprocessed_V1.csv"
THRESHOLD = 0.70

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_name = "MBZUAI-Paris/Atlas-Chat-9B"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
model.eval()

YES_TOKEN = tokenizer.encode("نعم", add_special_tokens=False)[0]
NO_TOKEN  = tokenizer.encode("لا", add_special_tokens=False)[0]

def get_yes_prob(prompt):
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    probs = F.softmax(torch.tensor([logits[YES_TOKEN], logits[NO_TOKEN]]), dim=0)
    return round(probs[0].item(), 3)

def detect_shhal(context):
    prompt = f"""النص: {context}

واش كاين في هذا النص عدد أو كمية أو رقم يمكن نسألو عليه بسؤال "شحال"؟
الجواب (نعم/لا):"""
    return get_yes_prob(prompt)

def detect_shno(context):
    prompt = f"""النص: {context}

واش كاين في هذا النص حدث أو قرار أو فعل يمكن نسألو عليه بسؤال "شنو وقع" أو "شنو قرر"؟
الجواب (نعم/لا):"""
    return get_yes_prob(prompt)

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

    for text in tqdm(df_slice["first_paragraph"].astype(str)):
        if text in processed_texts:
            continue

        shhal_prob = detect_shhal(text)
        shno_prob  = detect_shno(text)

        has_shhal = 1 if shhal_prob >= THRESHOLD else 0
        has_shno  = 1 if shno_prob  >= THRESHOLD else 0

        task = {
            "data": {
                "context": text,
                "has_shhal": has_shhal,
                "has_shno": has_shno
            }
        }

        tasks.append(task)

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)