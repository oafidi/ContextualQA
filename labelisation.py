from airllm import AutoModel
import json, re

model = AutoModel.from_pretrained(
    "MBZUAI-Paris/Atlas-Chat-9B",
    compression='4bit'
)

def generate_qa(context):
    prompt = f"""أعطيني سؤال وجواب واحد على هذا النص بالداريجة.
القواعد:
- السؤال لازم يكون بالداريجة
- الجواب لازم يكون مقتطف حرفي من النص
- ما تزيدش معلومات من برا النص
- رد فقط بـ JSON: {{"question": "...", "answer": "..."}}

النص: {context}"""

    tokens = model.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    output = model.generate(
        tokens['input_ids'],
        max_new_tokens=150,
        temperature=0.3,
        do_sample=True,
    )

    response = model.tokenizer.decode(output[0], skip_special_tokens=True)
    response = response[len(prompt):]

    try:
        match = re.search(r'\{.*?\}', response, re.DOTALL)
        if match:
            result = json.loads(match.group())
            result["valid"] = result.get("answer", "") in context
            return result
    except:
        pass

    return {"raw_response": response, "valid": False}

# Test on one example
context = "نظمت قبيلة الرگيبات البيهات لقاء تواصليا مساء الأحد الموافق لتاريخ 16 نونبر 2025 بمدينة السمارة."
print(generate_qa(context))