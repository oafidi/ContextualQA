import time
import json
import csv
import subprocess
import logging
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GEMINI_URL  = "https://chatgpt.com/"
OUTPUT_FILE = Path("./darija_questions.json")
LOG_FILE    = Path("./scraper.log")
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────
PROMPT_TEMPLATE = """" نتا خبير ف الداريجة المغربية .

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

# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
ALLOWED_STARTS   = ["شكون", "فين", "فوقاش", "شنو", "شحال"]
FORBIDDEN_STARTS = ["علاش", "كيفاش"]
 
def is_valid_question(text: str) -> bool:
    q = text.strip()
    if not q.endswith("؟"):
        return False
    starts_ok  = any(q.startswith(w) for w in ALLOWED_STARTS)
    starts_bad = any(q.startswith(w) for w in FORBIDDEN_STARTS)
    return starts_ok and not starts_bad
 
def clean_question(text: str) -> str:
    """Extrait la question depuis la réponse brute de Gemini."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return ""
    for line in reversed(lines):
        if "؟" in line:
            return line[:line.index("؟") + 1].strip()
    return lines[-1]
 
# ─────────────────────────────────────────────
# APPEND TO JSON
# ─────────────────────────────────────────────
def append_to_json(entry: dict):
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
 
    data.append(entry)
 
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
 
    log.info(f"Sauvegarde OK ({len(data)} entrees au total)")
 
# ─────────────────────────────────────────────
# CLIPBOARD (fix pour texte arabe)
# ─────────────────────────────────────────────
def copy_to_clipboard(text: str):
    """Copie le texte dans le clipboard Windows via PowerShell."""
    escaped = text.replace("'", "''")
    cmd = f"Set-Clipboard -Value '{escaped}'"
    subprocess.run(
        ["powershell", "-command", cmd],
        check=True,
        capture_output=True
    )
 
# ─────────────────────────────────────────────
# SELENIUM
# ─────────────────────────────────────────────
def create_driver(profile_path: str = None) -> webdriver.Chrome:
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--window-size=1280,900")
    if profile_path:
        options.add_argument(f"--user-data-dir={profile_path}")
    driver = webdriver.Chrome(options=options)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver
 
 
def wait_for_response(driver: webdriver.Chrome, timeout: int = 90) -> str:
    """Attend la fin de génération Gemini et retourne le texte."""
    time.sleep(3)
    end_time = time.time() + timeout
 
    while time.time() < end_time:
        try:
            driver.find_element(By.CSS_SELECTOR, "[aria-label='Stop generating']")
            time.sleep(1)
        except NoSuchElementException:
            time.sleep(1.5)
            break
 
    selectors = [
        "model-response .response-content",
        ".model-response-text",
        "message-content",
        "[data-response-index]",
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
 
    try:
        blocks = driver.find_elements(
            By.CSS_SELECTOR, "[role='article'], div[class*='response']"
        )
        if blocks:
            return blocks[-1].text.strip()
    except Exception:
        pass
 
    return ""
 
 
def send_prompt(driver: webdriver.Chrome, prompt: str) -> str:
    """Envoie un prompt via clipboard (supporte arabe + caractères spéciaux)."""
 
    # Copier dans le clipboard Windows
    copy_to_clipboard(prompt)
    time.sleep(0.5)
 
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
        raise Exception("Zone de texte Gemini introuvable")
 
    input_box.click()
    time.sleep(0.3)
    input_box.send_keys(Keys.CONTROL + "a")
    input_box.send_keys(Keys.DELETE)
    time.sleep(0.2)
 
    # Coller depuis le clipboard
    input_box.send_keys(Keys.CONTROL + "v")
    time.sleep(1.0)
 
    input_box.send_keys(Keys.RETURN)
    log.info("Prompt colle et envoye, attente de la reponse...")
 
    return wait_for_response(driver)
 
 
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    import argparse
 
    parser = argparse.ArgumentParser(
        description="Generateur de questions Darija depuis CSV"
    )
    parser.add_argument("--csv",     "-c", required=True,
                        help="Fichier CSV avec colonne 'first_paragraph'")
    parser.add_argument("--delay",   "-d", type=int, default=6,
                        help="Delai entre requetes en secondes (defaut: 6)")
    parser.add_argument("--profile", "-p", default=None,
                        help="Chemin profil Chrome")
    parser.add_argument("--limit",   "-l", type=int, default=None,
                        help="Limiter a N contextes (pour tester)")
    parser.add_argument("--retry",         type=int, default=2,
                        help="Tentatives si question invalide (defaut: 2)")
    args = parser.parse_args()
 
    # ── Lire le CSV ──────────────────────────
    contexts = []
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("first_paragraph", "").strip()
            if text:
                contexts.append(text)
 
    if args.limit:
        contexts = contexts[:args.limit]
 
    log.info(f"{len(contexts)} contextes charges depuis {args.csv}")
 
    # ── Démarrer Chrome ───────────────────────
    driver = create_driver(args.profile)
    driver.get(GEMINI_URL)
    time.sleep(5)
 
    if "accounts.google.com" in driver.current_url or "signin" in driver.current_url:
        log.warning("Connectez-vous a Google dans le navigateur ouvert...")
        input("Appuyez sur Entree apres connexion: ")
 
    log.info("Connecte a Gemini")
 
    # ── Boucle principale ─────────────────────
    success = 0
    skipped = 0
 
    for i, context in enumerate(contexts, 1):
        log.info(f"[{i}/{len(contexts)}] Traitement en cours...")
 
        question = None
 
        for attempt in range(1, args.retry + 2):
            try:
                raw = send_prompt(driver, PROMPT_TEMPLATE.format(context=context))
            except Exception as e:
                log.error(f"Erreur: {e}")
                break
 
            candidate = clean_question(raw)
 
            if is_valid_question(candidate):
                question = candidate
                log.info(f"Question valide: {question}")
                break
            else:
                log.warning(
                    f"Tentative {attempt} invalide: '{candidate[:60]}' — retry..."
                )
                time.sleep(3)
 
        if question:
            append_to_json({"context": context, "question": question})
            success += 1
        else:
            log.error(f"Contexte {i} ignore apres {args.retry + 1} tentatives")
            skipped += 1
 
        if i < len(contexts):
            time.sleep(args.delay)
 
    driver.quit()
    log.info("=" * 50)
    log.info(f"Termine: {success} generes, {skipped} ignores")
    log.info(f"Fichier: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
 