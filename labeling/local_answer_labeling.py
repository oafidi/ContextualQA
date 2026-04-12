from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json, os
from tqdm import tqdm

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
INPUT_JSON  = "datasets/processed/question_generated_by_gemini_scraper/questions_7626_7863.json"
OUTPUT_JSON = "datasets/processed/answers/answers_7626_7863.json"

# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_name = "MBZUAI-Paris/Atlas-Chat-2B"  
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)



# ─────────────────────────────────────────────
#  POST-PROCESSING
#  Valide que la réponse est un extrait verbatim
#  du contexte. Si le modèle a ajouté du texte
#  en trop, on tente de récupérer le plus long
#  préfixe valide.
# ─────────────────────────────────────────────
def extract_valid_span(raw: str, context: str) -> str | None:
    raw = raw.split("\n")[0].strip()
    raw = raw.strip("\"'«»""''")

    # Cas idéal : la réponse entière est dans le contexte
    if raw and raw in context:
        return raw

    # Tenter de trouver le plus long préfixe valide
    for end in range(len(raw), 0, -1):
        candidate = raw[:end].strip()
        if candidate and candidate in context:
            return candidate

    return None


# ─────────────────────────────────────────────
#  ANSWER GENERATION — PROMPT ENGINEERING ONLY
# ─────────────────────────────────────────────
def generate_answer(context: str, question: str) -> str | None:

    # Few-shot examples : montrer exactement ce qu'on attend
    prompt = (
        "استخرج الجواب من النص كلمة بكلمة، بلا ما تزيد حتى حاجة من عندك.\n"
        "الجواب خاصو يكون نسخة مطابقة من النص.\n\n"

        "─────────────────\n"
        "مثال 1\n"
        'النص: """افتتح الملك محمد السادس، يوم الاثنين بالرباط، المعرض الدولي للنشر والكتاب في دورته الثلاثين."""\n'
        "السؤال: فين افتتح الملك المعرض الدولي للنشر والكتاب؟\n"
        "الجواب: بالرباط\n\n"

        "─────────────────\n"
        "مثال 2\n"
        'النص: """أعلنت وزارة الصحة عن إطلاق حملة تلقيح وطنية تستهدف الأطفال دون سن الخامسة."""\n'
        "السؤال: شكون علن على حملة التلقيح؟\n"
        "الجواب: وزارة الصحة\n\n"

        "─────────────────\n"
        "مثال 3\n"
        'النص: """فاز المنتخب المغربي بكأس أمم أفريقيا للمرة الثانية في تاريخه."""\n'
        "السؤال: شحال من مرة فاز المنتخب المغربي بكأس أمم أفريقيا؟\n"
        "الجواب: الثانية\n\n"

        "─────────────────\n"
        f'النص: """{context}"""\n'
        f"السؤال: {question}\n"
        "الجواب:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1536,
    ).to(model.device)

    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens = 40,
            do_sample      = False,
            pad_token_id   = tokenizer.eos_token_id,
        )

    generated_ids = output[0][prompt_length:]
    raw_answer    = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return extract_valid_span(raw_answer, context)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    print(f"{len(tasks)} questions chargées")

    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"{len(results)} réponses déjà générées")
    else:
        results = []

    processed_contexts = {r["data"]["context"] for r in results}

    accepted = 0
    rejected = 0

    for task in tqdm(tasks):
        context  = task["data"]["context"]
        question = task["data"]["question"]

        if not question or context in processed_contexts:
            continue

        answer = generate_answer(context, question)

        if answer is not None:
            accepted += 1
        else:
            rejected += 1

        results.append({
            "data": {
                "context" : context,
                "question": question,
                "answer"  : answer,
            }
        })

        # Sauvegarde incrémentale
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Sauvegarde finale
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total = accepted + rejected
    if total > 0:
        print(f"\nTerminé — {total} traités")
        print(f"   Acceptées : {accepted} ({accepted / total * 100:.1f}%)")
        print(f"   Rejetées  : {rejected} ({rejected / total * 100:.1f}%)")
    else:
        print("\nTerminé — rien à traiter")
    print(f"   Sauvegardé → {OUTPUT_JSON}")