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

model_name = "MBZUAI-Paris/Atlas-Chat-9B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

FALLBACK_QUESTIONS = "/ شكون / فين / فوقاش / معامن / شنو / شحال /"

def generate_q(context, questions):
    context_tokens = tokenizer(context, truncation=True, max_length=400)
    context = tokenizer.decode(context_tokens["input_ids"], skip_special_tokens=True)

    prompt = f"""نتا خبير ف الداريجة المغربية. مهمتك هي تكتب سؤال واحد بالداريجة جوابو موجود صريح فالنص.

--- مثال 1 ---
النص: "فاز الرجاء البيضاوي بكأس العرش بعد ما هزم الوداد بهدف وحيد سجله بوفال فالدقيقة 78 فمدينة فاس."
السؤال: فين تلاقاو الرجاء والوداد فنهائي كأس العرش؟

--- مثال 2 ---
النص: "أعلنت وزارة الصحة المغربية على توصيل 3 ملايين جرعة من اللقاح خلال شهر يناير 2024، وهادشي جعل المغرب يتصدر قائمة دول إفريقيا فحملات التلقيح."
السؤال: شحال من جرعة وصلات فالمغرب خلال شهر يناير 2024؟

--- مثال 3 ---
النص: "صرح المدير العام للمكتب الشريف للفوسفاط، مصطفى التراب، بأن رقم المعاملات وصل 9 مليار دولار فالسنة الماضية."
السؤال: شكون هو المدير العام ديال المكتب الشريف للفوسفاط؟

--- دابا نتا ---
النص: \"\"\"{context}\"\"\"

كتب سؤال واحد يبدا بـ: {questions}
- الجواب خاصو يكون موجود صريح فالنص
- جملة واحدة تنتهي بـ؟
- بلا أي مقدمة أو شرح

السؤال:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    question = response.split("\n")[0].strip()

    if not question.endswith("؟") or len(question) < 10:
        question = None

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
        "has_shno"     : "/ شنو /",
        "has_shhal"    : "/ شحال /",
        "has_location" : "/ فين /",
        "has_date"     : "/ فوقاش / امتا / وقتاش / معاش /",
        "has_person"   : "/ شكون / معامن /"
    }
    columns = ["has_shno", "has_shhal", "has_location", "has_date", "has_person"]

    for index, row in tqdm(df_slice.iterrows(), total=len(df_slice)):

        context = row["context"]

        if context in processed_texts:
            continue

        # Build question types string
        questions = ""
        i = 0
        for column in columns:
            if int(row[column]) == 1:
                i += 1
                questions += question_types[column]

        questions = questions.replace("//", "/")

        # Fallback
        if i <= 1:
            questions = FALLBACK_QUESTIONS
        elif int(row["has_shhal"]) == 1 and int(row["has_date"]) == 1 and int(row["has_shno"]) == 1:
            questions = "/ فوقاش / امتا / معاش / شحال /"
        elif int(row["has_shhal"]) == 1 and int(row["has_shno"]) == 1:
            questions = "شحال"
        elif int(row["has_date"]) == 1 and int(row["has_shno"]) == 1:
            questions = "/ فوقاش / امتا / وقتاش / معاش /"
        
        generated_question = generate_q(context, questions)

        entry = {
            "data": {
                "context"       : context,
                "question"      : generated_question,
            }
        }
        tasks.append(entry)

        # Flush every n entries
        if len(tasks) % 5 == 0:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(tasks, f, ensure_ascii=False, indent=2)

    # Final save
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
