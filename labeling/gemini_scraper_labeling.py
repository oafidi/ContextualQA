"""
=======================
Automated Darija (Moroccan Arabic) QA pair generator using Google Gemini.

CHANGES FROM ORIGINAL:
    1. Generates both a QUESTION and an ANSWER for each context.
       The answer must be an exact substring of the context (extractive QA style).
    2. Every CHAT_RESET_EVERY contexts, the script opens a brand-new Gemini chat
       tab — this prevents conversation history from influencing future answers,
       avoids session-level rate-limits, and mimics human browsing patterns.
    3. Gemini is now asked to reply in strict JSON so the answer field can be
       parsed and validated reliably without regex.

HOW IT WORKS (big picture):
    1. Read text passages ("contexts") from a CSV file.
    2. Open a real Chrome browser window using Selenium.
    3. For each context, paste a carefully crafted Arabic prompt into Gemini's
       chat box and press Enter.
    4. Parse the JSON response to extract (question, answer).
    5. Validate:
         - question starts with an allowed Darija interrogative word and ends with ؟
         - answer is a non-empty exact substring of the context
    6. Every CHAT_RESET_EVERY contexts, close the current chat and start a new one
       by navigating to a fresh Gemini URL.
    7. Save every valid (context, question, answer) triple into a JSON file
       immediately, so no work is lost if the script crashes.

ANTI-BAN STRATEGY (unchanged from original):
    - Random pauses between actions and requests.
    - Occasional page scrolling.
    - Realistic browser window size and user-agent string.
    - Selenium automation fingerprints are hidden.
    - CAPTCHA and rate-limit detection with manual-intervention prompts.
    - Long break every N requests.
"""

import time
import json
import subprocess
import logging
import random
import platform
from pathlib import Path

import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException


# =============================================================================
# CONFIG
# =============================================================================

GEMINI_URL   = "https://gemini.google.com/app"

START_INDEX  = 350
END_INDEX    = 1500

# After this many *successfully processed* (or attempted) contexts, the script
# navigates to a fresh Gemini chat to reset the conversation history.
CHAT_RESET_EVERY = 5

OUTPUT_FILE  = Path(f"./datasets/qa_rest_part_1_{START_INDEX + 1}_{END_INDEX}.json")
LOG_FILE     = Path("./logs/scraper_qa.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================
# The prompt asks Gemini to output ONLY a valid JSON object with two fields:
#   "question" : a single Darija question ending with ؟
#   "answer"   : the exact substring from the context that answers the question
#
# Asking for JSON makes the answer trivially parseable and avoids free-form text
# that is hard to validate or extract.

PROMPT_TEMPLATE = """نتا خبير ف الداريجة المغربية. مهمتك هي تكتب سؤال واحد بالداريجة والجواب ديالو مباشرة من النص.

القواعد:
- السؤال خاصو يبدا بـ: شكون / فين / فوقاش / معامن / وقتاش / شنو / شحال
- السؤال خاصو ينتهي بـ ؟
- الجواب خاصو يكون نفس العبارة الموجودة فالنص بدون تغيير (مقتطف حرفي)
- الجواب ما خاصوش يكون جملة كاملة — عبارة قصيرة كافية
- الرد خاصو يكون JSON فقط، بلا أي نص إضافي

--- مثال 1 ---
النص: فاز الرجاء البيضاوي بكأس العرش بعد ما هزم الوداد بهدف وحيد سجله بوفال فالدقيقة 78 فمدينة فاس.
الرد:
{{"question": "فين لعبات الرجاء والوداد نهائي كأس العرش؟", "answer": "مدينة فاس"}}

--- مثال 2 ---
النص: أعلنت وزارة الصحة المغربية على توصيل 3 ملايين جرعة من اللقاح خلال شهر يناير 2024.
الرد:
{{"question": "شحال من جرعة لقاح وصلات المغرب خلال شهر يناير 2024؟", "answer": "3 ملايين جرعة"}}

--- مثال 3 ---
النص: صرح المدير العام للمكتب الشريف للفوسفاط، مصطفى التراب، بأن رقم المعاملات وصل 9 مليار دولار.
الرد:
{{"question": "شكون هو المدير العام ديال المكتب الشريف للفوسفاط؟", "answer": "مصطفى التراب"}}

--- مثال 4 ---
النص: استقبل الرئيس الكولومبي گوستافو بيترو، يوم الخميس 27 نونبر 2025، مسؤول البوليساريو لأمريكا اللاتينية محمد زروگ.
الرد:
{{"question": "فوقاش استقبل الرئيس الكولومبي مسؤول البوليساريو؟", "answer": "يوم الخميس 27 نونبر 2025"}}

--- مثال 5 ---
النص: قال أمين عام حزب الله نعيم قاسم إن قرار الحكومة اللبنانية تجريد الحزب من سلاحه قد يؤدي إلى حرب أهلية.
الرد:
{{"question": "شنو قال نعيم قاسم فخصوص قرار تجريد حزب الله من سلاحه؟", "answer": "قد يؤدي إلى حرب أهلية"}}

--- دابا نتا ---
النص: {context}

الرد (JSON فقط):"""


# =============================================================================
# VALIDATION
# =============================================================================

ALLOWED_STARTS  = ["شكون", "فين", "فوقاش", "شنو", "شحال", "معامن", "وقتاش"]
FORBIDDEN_STARTS = ["علاش", "كيفاش"]


def is_valid_question(text: str) -> bool:
    """Return True if `text` is a well-formed Darija question."""
    q = text.strip()
    if not q.endswith("؟"):
        return False
    starts_ok  = any(q.startswith(w) for w in ALLOWED_STARTS)
    starts_bad = any(q.startswith(w) for w in FORBIDDEN_STARTS)
    return starts_ok and not starts_bad


def is_valid_answer(answer: str, context: str) -> bool:
    """
    Return True if `answer` is a non-empty exact substring of `context`.

    This enforces extractive (span-selection) QA: the answer must literally
    appear in the passage — no paraphrasing or hallucination allowed.
    """
    answer = answer.strip()
    return bool(answer) and answer in context


def parse_qa_response(raw: str, context: str):
    """
    Parse Gemini's raw response and return (question, answer) or (None, None).

    Gemini is instructed to respond with only a JSON object.  However it
    sometimes wraps the JSON in markdown fences (```json ... ```) or adds a
    short preamble.  This function is tolerant of both.

    Returns:
        (question_str, answer_str)  — if both fields are valid
        (None, None)                — if parsing fails or validation fails
    """
    # Strip markdown code fences if present
    text = raw.strip()
    for fence in ("```json", "```"):
        if fence in text:
            # Take the content between the first pair of fences
            parts = text.split(fence)
            # parts[1] is the content between opening and closing fence
            if len(parts) >= 3:
                text = parts[1].strip()
                break
            elif len(parts) == 2:
                text = parts[1].strip()
                break

    # Sometimes Gemini outputs multiple lines before the JSON — find the first '{'
    brace_start = text.find("{")
    brace_end   = text.rfind("}")
    if brace_start == -1 or brace_end == -1 or brace_end <= brace_start:
        log.warning(f"No JSON object found in response: {raw[:120]}")
        return None, None

    json_str = text[brace_start : brace_end + 1]

    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error ({e}): {json_str[:120]}")
        return None, None

    question = str(obj.get("question", "")).strip()
    answer   = str(obj.get("answer",   "")).strip()

    if not is_valid_question(question):
        log.warning(f"Invalid question format: '{question[:80]}'")
        return None, None

    if not is_valid_answer(answer, context):
        log.warning(
            f"Answer not found in context.\n"
            f"  answer : '{answer[:80]}'\n"
            f"  context: '{context[:80]}'"
        )
        return None, None

    return question, answer


# =============================================================================
# JSON PERSISTENCE — efficient append without reloading the whole file
# =============================================================================

def append_to_json(entry: dict):
    """
    Append a single dict entry to the JSON array stored in OUTPUT_FILE.
    Uses a binary seek trick to avoid rewriting the whole file each time.
    """
    new_entry_str = json.dumps(entry, ensure_ascii=False, indent=2)

    if not OUTPUT_FILE.exists():
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("[\n" + new_entry_str + "\n]")
        log.info("JSON file created with first entry")
        return

    with open(OUTPUT_FILE, "rb+") as f:
        f.seek(0, 2)
        pos = f.tell() - 1

        while pos > 0:
            f.seek(pos)
            ch = f.read(1)
            if ch == b"]":
                break
            pos -= 1

        f.seek(0)
        content_start = f.read(10).decode("utf-8").strip()
        is_empty_array = content_start in ["[]", "[\n]", "[ ]"]

        f.seek(pos)
        if is_empty_array:
            f.write(("\n" + new_entry_str + "\n]").encode("utf-8"))
        else:
            f.write((",\n" + new_entry_str + "\n]").encode("utf-8"))
        f.truncate()

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            count = len(json.load(f))
        log.info(f"Save successful ({count} total entries)")
    except Exception:
        log.info("Save successful")


# =============================================================================
# CLIPBOARD — cross-platform Arabic text workaround
# =============================================================================

def copy_to_clipboard(text: str):
    """Copy `text` to the system clipboard (OS-aware)."""
    current_os = platform.system()

    if current_os == "Windows":
        escaped = text.replace("'", "''")
        subprocess.run(
            ["powershell", "-command", f"Set-Clipboard -Value '{escaped}'"],
            check=True, capture_output=True
        )
    elif current_os == "Darwin":
        subprocess.run("pbcopy", input=text.encode("utf-8"), check=True)
    elif current_os == "Linux":
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.encode("utf-8"), check=True
        )
    else:
        log.warning(f"Unknown OS '{current_os}' — trying PowerShell as clipboard fallback")
        escaped = text.replace("'", "''")
        subprocess.run(
            ["powershell", "-command", f"Set-Clipboard -Value '{escaped}'"],
            check=True, capture_output=True
        )


# =============================================================================
# HUMAN BEHAVIOUR SIMULATION
# =============================================================================

def human_pause(min_s: float, max_s: float):
    time.sleep(random.uniform(min_s, max_s))


def human_scroll(driver):
    scroll_px = random.randint(80, 350)
    driver.execute_script(f"window.scrollBy(0, {scroll_px})")
    human_pause(0.3, 0.9)


def check_for_captcha(driver) -> bool:
    indicators = [
        "recaptcha", "captcha", "unusual traffic",
        "verify you're human", "تحقق", "Je ne suis pas un robot"
    ]
    page_text = driver.page_source.lower()
    return any(ind.lower() in page_text for ind in indicators)


def check_for_rate_limit(driver) -> bool:
    indicators = [
        "too many requests", "rate limit", "quota exceeded",
        "try again", "Something went wrong"
    ]
    page_text = driver.page_source.lower()
    return any(ind.lower() in page_text for ind in indicators)


# =============================================================================
# SELENIUM DRIVER SETUP
# =============================================================================

def create_driver(profile_path: str = None, profile_dir: str = "Default") -> webdriver.Chrome:
    """Create and return a configured Selenium Chrome WebDriver instance."""
    options = Options()

    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    w = random.choice([1280, 1366, 1440, 1536, 1920])
    h = random.choice([768, 800, 864, 900, 1080])
    options.add_argument(f"--window-size={w},{h}")

    current_os = platform.system()
    log.info(f"Detected OS: {current_os}")

    if current_os == "Windows":
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    elif current_os == "Darwin":
        user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    elif current_os == "Linux":
        user_agent = (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    else:
        log.warning(f"Unknown OS '{current_os}' — falling back to Windows user-agent")
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )

    options.add_argument(f"--user-agent={user_agent}")

    if profile_path:
        options.add_argument(f"--user-data-dir={profile_path}")
        options.add_argument(f"--profile-directory={profile_dir}")

    driver = webdriver.Chrome(options=options)

    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    driver.execute_cdp_cmd("Network.setUserAgentOverride", {"userAgent": user_agent})

    return driver


# =============================================================================
# CHAT RESET — open a fresh Gemini conversation
# =============================================================================

def reset_chat(driver: webdriver.Chrome):
    """
    Navigate to a brand-new Gemini chat by loading the base URL.

    WHY THIS IS IMPORTANT:
      Gemini carries the full conversation history in its context window.
      After many turns, this history can:
        - Bias the model toward certain phrasing styles it has already used.
        - Accumulate tokens until the context window limit is hit.
        - Cause increasingly incoherent or recycled responses.

      Opening a fresh chat URL gives Gemini a clean slate every CHAT_RESET_EVERY
      contexts, ensuring consistent, independent generation for each passage.

    The function adds a longer-than-usual delay after navigation to let the
    page fully reinitialise before the next prompt is sent.
    """
    log.info("=== Resetting chat — opening a new Gemini conversation ===")
    driver.get(GEMINI_URL)
    human_pause(4, 7)   # wait for the fresh page to load completely

    # If the reset landed us on a login page, prompt the user to re-authenticate.
    if "accounts.google.com" in driver.current_url or "signin" in driver.current_url:
        log.warning("Session expired after chat reset — please log in again...")
        input("Press Enter after signing in: ")


# =============================================================================
# WAITING FOR GEMINI'S RESPONSE
# =============================================================================

def wait_for_response(driver: webdriver.Chrome, timeout: int = 90) -> str:
    """Wait until Gemini finishes generating and return the response text."""
    human_pause(15, 20)
    end_time = time.time() + timeout

    # Phase 1: wait for the "Stop generating" button to disappear
    while time.time() < end_time:
        try:
            driver.find_element(By.CSS_SELECTOR, "[aria-label='Stop generating']")
            time.sleep(1)
        except NoSuchElementException:
            human_pause(1, 2)
            break

    # Phase 2: extract the last response element
    selectors = [
        "model-response .response-content",
        ".model-response-text",
        "message-content",
        "[data-response-index]",
        ".markdown",
    ]

    for sel in selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, sel)
            if elements:
                text = elements[-1].text.strip()
                if text:
                    return text
        except Exception:
            continue

    # Phase 3: fallback selectors
    try:
        blocks = driver.find_elements(
            By.CSS_SELECTOR, "[role='article'], div[class*='response']"
        )
        if blocks:
            return blocks[-1].text.strip()
    except Exception:
        pass

    return ""


# =============================================================================
# SENDING THE PROMPT TO GEMINI
# =============================================================================

def send_prompt(driver: webdriver.Chrome, prompt: str) -> str:
    """Paste the prompt into Gemini's input box and return the raw response."""
    copy_to_clipboard(prompt)
    human_pause(0.4, 0.8)

    input_selectors = [
        "div[contenteditable='true'][role='textbox']",
        "rich-textarea div[contenteditable]",
        "textarea",
    ]

    input_box = None
    for sel in input_selectors:
        try:
            input_box = WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
            )
            break
        except TimeoutException:
            continue

    if not input_box:
        raise Exception("Gemini text input box not found — page may have changed")

    input_box.click()
    human_pause(0.2, 0.5)

    input_box.send_keys(Keys.CONTROL + "a")
    human_pause(0.1, 0.3)
    input_box.send_keys(Keys.DELETE)
    human_pause(0.2, 0.4)

    input_box.send_keys(Keys.CONTROL + "v")
    human_pause(0.8, 1.5)

    input_box.send_keys(Keys.RETURN)
    log.info("Prompt submitted — waiting for response...")

    return wait_for_response(driver)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Entry point. Reads the CSV, drives Chrome, generates QA pairs, saves JSON.

    NEW CLI FLAGS (compared to original):
      --chat-reset-every   Override CHAT_RESET_EVERY at runtime (default: 5).

    CHAT RESET LOGIC:
      A counter `contexts_since_reset` tracks how many contexts have been
      processed (or attempted) since the last chat reset.  When it reaches
      CHAT_RESET_EVERY, reset_chat() is called before the next request and
      the counter is reset to 0.  This happens regardless of whether the
      previous contexts succeeded or failed validation.

    OUTPUT FORMAT (one entry):
      {
        "data": {
          "context" : "...",
          "question": "...",
          "answer"  : "..."   ← exact substring of context
        }
      }
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Darija QA pair generator — cross-platform, anti-ban, with chat reset"
    )
    parser.add_argument("--csv",             "-c", required=True,
                        help="CSV file path (must contain a 'context' column)")
    parser.add_argument("--delay",           "-d", type=int, default=8,
                        help="Min delay between requests in seconds (default: 8)")
    parser.add_argument("--profile",         "-p", default=None,
                        help="Chrome user-data-dir path for pre-authenticated profile")
    parser.add_argument("--profile-dir",           default="Default",
                        help="Chrome profile folder name (e.g. 'Profile 2')")
    parser.add_argument("--retry",                 type=int, default=2,
                        help="Extra attempts if response is invalid (default: 2)")
    parser.add_argument("--pause-every",           type=int, default=20,
                        help="Long break every N requests (default: 20)")
    parser.add_argument("--chat-reset-every",      type=int, default=CHAT_RESET_EVERY,
                        help=f"Open a new chat every N contexts (default: {CHAT_RESET_EVERY})")
    args = parser.parse_args()

    chat_reset_every = args.chat_reset_every

    # ── Read contexts ─────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv, encoding="utf-8")
    df_slice = df.iloc[START_INDEX:END_INDEX]
    log.info(f"Processing rows {START_INDEX}–{END_INDEX - 1} ({len(df_slice)} contexts)")

    # ── Resume support ────────────────────────────────────────────────────────
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
        log.info(f"Loaded {len(existing)} already-processed entries")
    else:
        existing = []

    processed_texts = {task["data"]["context"] for task in existing}

    # ── Launch Chrome ─────────────────────────────────────────────────────────
    driver = create_driver(args.profile, args.profile_dir)
    driver.get(GEMINI_URL)
    human_pause(4, 7)

    if "accounts.google.com" in driver.current_url or "signin" in driver.current_url:
        log.warning("Not signed in — please log in in the browser window...")
        input("Press Enter after signing in: ")

    log.info("Connected to Gemini — starting processing loop")

    # ── Main loop ─────────────────────────────────────────────────────────────
    success               = 0
    skipped               = 0
    already_done          = 0
    contexts_since_reset  = 0   # tracks contexts processed since last chat reset

    for loop_pos, (csv_index, row) in enumerate(df_slice.iterrows(), start=1):

        context_text = str(row["context"]).strip()

        if context_text in processed_texts:
            already_done += 1
            log.info(f"[{loop_pos}] Row #{csv_index} already processed — skipping")
            continue

        # ── Chat reset check ──────────────────────────────────────────────────
        # Reset *before* the request if we've hit the threshold.
        # Using >= instead of == ensures we reset even if the counter was somehow
        # incremented past the threshold (e.g. due to retries in the same slot).
        if contexts_since_reset > 0 and contexts_since_reset % chat_reset_every == 0:
            reset_chat(driver)
            contexts_since_reset = 0   # counter resets after opening new chat

        log.info(
            f"[{loop_pos}] Processing row #{csv_index} "
            f"(chat turn {contexts_since_reset + 1}/{chat_reset_every})..."
        )

        # ── Safety checks ─────────────────────────────────────────────────────
        if check_for_captcha(driver):
            log.warning("CAPTCHA detected — please solve it manually...")
            input("Press Enter after solving the CAPTCHA: ")

        if check_for_rate_limit(driver):
            wait_time = random.randint(5, 10)
            log.warning(f"Rate limit detected — waiting {wait_time}s then refreshing...")
            time.sleep(wait_time)
            driver.refresh()
            human_pause(3, 6)

        if loop_pos > 1 and loop_pos % args.pause_every == 0:
            pause = random.randint(1, 5)
            log.info(f"Long pause ({pause}s) after {loop_pos} requests...")
            time.sleep(pause)
            human_scroll(driver)

        human_scroll(driver)

        # ── Generate QA pair with retries ─────────────────────────────────────
        question = None
        answer   = None

        for attempt in range(1, args.retry + 2):
            try:
                raw = send_prompt(
                    driver,
                    PROMPT_TEMPLATE.format(context=context_text)
                )
            except Exception as e:
                log.error(f"Error sending prompt: {e}")
                break

            q, a = parse_qa_response(raw, context_text)

            if q and a:
                question = q
                answer   = a
                log.info(f"Question : {question}")
                log.info(f"Answer   : {answer}")
                break
            else:
                log.warning(f"Attempt {attempt} failed — retrying in 3–6 s...")
                human_pause(3, 6)

        # ── Persist result ────────────────────────────────────────────────────
        if question and answer:
            entry = {
                "data": {
                    "context" : context_text,
                    "question": question,
                    "answer"  : answer,
                }
            }
            append_to_json(entry)
            success += 1
        else:
            log.error(
                f"Row #{csv_index} skipped after {args.retry + 1} failed attempts"
            )
            skipped += 1

        # Increment AFTER the attempt (success or failure) so the reset fires
        # correctly at the start of the *next* iteration.
        contexts_since_reset += 1

        delay = random.uniform(args.delay, args.delay * 2)
        log.info(f"Waiting {delay:.1f}s before next request...")
        time.sleep(delay)

    # ── Summary ───────────────────────────────────────────────────────────────
    driver.quit()
    log.info(
        f"Done — generated: {success}, failed: {skipped}, already done: {already_done}"
    )
    log.info(f"Output: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()