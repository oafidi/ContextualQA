import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import json, os

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
NO_TOKEN  = tokenizer.encode("لا",  add_special_tokens=False)[0]
THRESHOLD = 0.70
LOG_PATH  = "datasets/processed/qa_types_log.json"
INPUT_PATH = "datasets/processed/goud_preprocessed_V1.csv"
OUTPUT_PATH = "datasets/processed/goud_preprocessed_V2.csv"


def get_yes_prob(prompt):
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=512).to("cuda")
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

def detect_types(context, threshold=THRESHOLD):
    shhal_prob = detect_shhal(context)
    shno_prob  = detect_shno(context)
    return shhal_prob, shno_prob


def save_log(context, shhal_prob, shno_prob, threshold=THRESHOLD):
    entry = {
        "context"   : context,
        "score_shhal": shhal_prob,
        "score_shno" : shno_prob,
        "has_shhal"  : 1 if shhal_prob >= threshold else 0,
        "has_shno"   : 1 if shno_prob  >= threshold else 0,
    }

    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                log = json.load(f)
        except (json.JSONDecodeError, IOError):
            log = []
    else:
        log = []

    log.append(entry)

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":

    df = pd.read_csv(INPUT_PATH, encoding="utf-8")
    print(f"{len(df)} articles chargés")

    results = []
    for text in tqdm(df["first_paragraph"].astype(str)):
        shhal_prob, shno_prob = detect_types(text)
        save_log(text, shhal_prob, shno_prob)
        results.append((shhal_prob, shno_prob))

    df["score_shhal"] = [r[0] for r in results]
    df["score_shno"]  = [r[1] for r in results]
    df["has_shhal"]   = [1 if r[0] >= THRESHOLD else 0 for r in results]
    df["has_shno"]    = [1 if r[1] >= THRESHOLD else 0 for r in results]

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
