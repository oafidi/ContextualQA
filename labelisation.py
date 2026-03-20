from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json, os
import pandas as pd

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_name = "Qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)

def generate_q(context):
    prompt = f"""" نتا خبير ف الداريجة المغربية .

قرا هاد النص مزيان:
\"\"\"
{context}
\"\"\"

عطيني سؤال واحد بالداريجة:
- يبدا بـ: شكون / فين / فوقاش / شنو / شحال
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

def save_question(context, question, filepath="qa.json"):
    entry = {
        "context": context,
        "question": question
    }
    
    # Load existing data or start fresh
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = []
    else:
        data = []
    
    data.append(entry)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Question Saved ({len(data)} total entries)")

# Test

df = pd.read_csv("context.csv")

for i in range (50):
    context = df["context"].iloc[i]
    question = generate_q(context)
    if question:
        save_question(context, question)