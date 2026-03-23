"""
=======================
Automated Darija (Moroccan Arabic) question generator using Google Gemini.

HOW IT WORKS (big picture):
    1. Read text passages ("contexts") from a CSV file.
    2. Open a real Chrome browser window using Selenium (a browser automation library).
    3. For each context, paste a carefully crafted Arabic prompt into Gemini's chat box
       and press Enter — exactly as a human user would.
    4. Wait for Gemini to reply, then extract and validate the generated question.
    5. Save every valid (context, question) pair into a JSON file immediately,
       so no work is lost if the script crashes.

WHAT IS SELENIUM?
    Selenium is a Python library that lets you control a real web browser (Chrome here)
    from your code. It can find HTML elements on a page (buttons, text boxes...),
    click them, type text, read their content, and much more — all programmatically.
    It is widely used for web testing and web scraping.

ANTI-BAN STRATEGY:
    Google can detect and block bots. This script mimics human behaviour by:
    - Adding random pauses between actions and between requests.
    - Scrolling the page randomly.
    - Using a realistic browser window size and user-agent string.
    - Hiding Selenium's automation fingerprints (webdriver flag, etc.).
    - Detecting CAPTCHAs and rate-limit messages and pausing accordingly.
    - Taking a long break every N requests.

CROSS-PLATFORM:
    The script automatically detects the operating system (Windows, macOS, Linux)
    and applies the correct user-agent string and clipboard method for each.

USAGE:
    python darija_scraper_final.py --csv data.csv
    python darija_scraper_final.py --csv data.csv --delay 10 --pause-every 15
    python darija_scraper_final.py --csv data.csv --profile "C:/Users/you/AppData/Local/Google/Chrome/User Data"
"""

import time
import json
import subprocess
import logging
import random
import platform                  # used to detect the current operating system
from pathlib import Path

import pandas as pd              # used to read the CSV file

# --- Selenium imports ---
# webdriver            : the main class that controls the Chrome browser instance
# By                   : lets you locate HTML elements by CSS selector, ID, XPath, etc.
# Keys                 : keyboard key constants (Enter, Ctrl, Delete ...)
# WebDriverWait        : pauses execution until a condition is met (element appears, ...)
# EC                   : a library of pre-built wait conditions (element_to_be_clickable, ...)
# Options              : Chrome-specific startup options (window size, flags, profile ...)
# TimeoutException     : raised by WebDriverWait when the condition never becomes true
# NoSuchElementException : raised when find_element() cannot find the target element
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

# URL of the Gemini chat interface — this is what the browser will open.
GEMINI_URL = "https://gemini.google.com/app"

# Row range to process from the CSV (0-based, END_INDEX is excluded).
# Example: START_INDEX=1525, END_INDEX=3050 processes rows 1525 to 3049.
START_INDEX = 1525
END_INDEX   = 3050

# Path to the output JSON file where (context, question) pairs are saved.
OUTPUT_FILE = Path(f"./datasets/processed/question_generated_by_gemini_scraper/questions_{START_INDEX + 1}_{END_INDEX}.json")

# Path to the log file for persistent logging.
LOG_FILE = Path("./logs/scraper.log")

# Set up logging: messages go both to a file and to the terminal.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),   # writes to scraper.log
        logging.StreamHandler()                            # prints to the console
    ]
)
log = logging.getLogger(__name__)


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================
# This is the Arabic text we send to Gemini for each context.
# {context} is a placeholder replaced with the actual passage at runtime.
# The prompt instructs Gemini to:
#   - Produce exactly ONE question in Moroccan Darija.
#   - Start the question with one of the allowed interrogative words.
#   - Never use the forbidden question-starters.
#   - Ensure the answer is explicitly present in the provided text.
#   - Output nothing but the question itself (no preamble, no explanation).

PROMPT_TEMPLATE = """نتا خبير ف الداريجة المغربية. مهمتك هي تكتب سؤال واحد بالداريجة يكون الجواب ديالو موجود و صريح فالنص.

--- مثال 1 ---
النص: فاز الرجاء البيضاوي بكأس العرش بعد ما هزم الوداد بهدف وحيد سجله بوفال فالدقيقة 78 فمدينة فاس.
السؤال: فين لعبات الرجاء والوداد نهائي كأس العرش؟

--- مثال 2 ---
النص: أعلنت وزارة الصحة المغربية على توصيل 3 ملايين جرعة من اللقاح خلال شهر يناير 2024.
السؤال: شحال من جرعة لقاح وصلات المغرب خلال شهر يناير 2024؟

--- مثال 3 ---
النص: صرح المدير العام للمكتب الشريف للفوسفاط، مصطفى التراب، بأن رقم المعاملات وصل 9 مليار دولار.
السؤال: شكون هو المدير العام ديال المكتب الشريف للفوسفاط؟

--- مثال 4 ---
النص: استقبل الرئيس الكولومبي گوستافو بيترو، يوم الخميس 27 نونبر 2025، مسؤول البوليساريو لأمريكا اللاتينية محمد زروگ.
السؤال: فوقاش استقبل الرئيس الكولومبي مسؤول البوليساريو؟

--- مثال 5 ---
النص: قال أمين عام حزب الله نعيم قاسم إن قرار الحكومة اللبنانية تجريد الحزب من سلاحه قد يؤدي إلى حرب أهلية.
السؤال: شنو قال نعيم قاسم فخصوص السلاح؟

--- دابا نتا ---
النص: {context}

كتب سؤال واحد :
- يبدا بـ: شكون / فين / فوقاش / معامن / وقتاش / شنو / شحال
- الجواب خاصو يكون موجود صريح فالنص
- جملة واحدة تنتهي بـ؟
- بلا أي مقدمة أو شرح

السؤال:"""


# =============================================================================
# VALIDATION
# =============================================================================

# Questions MUST start with one of these Darija interrogative words.
ALLOWED_STARTS = ["شكون", "فين", "فوقاش", "شنو", "شحال", "معامن", "وقتاش"]

# Questions must NOT start with any of these words (why / how — excluded by design).
FORBIDDEN_STARTS = ["علاش", "كيفاش"]


def is_valid_question(text: str) -> bool:
    """
    Return True only when `text` meets all three criteria:
      1. Ends with the Arabic question mark '؟'.
      2. Starts with one of the allowed interrogative words.
      3. Does NOT start with a forbidden word.

    This prevents saving off-topic answers, explanations, or disallowed question types.
    """
    q = text.strip()
    if not q.endswith("؟"):
        return False
    starts_ok  = any(q.startswith(w) for w in ALLOWED_STARTS)
    starts_bad = any(q.startswith(w) for w in FORBIDDEN_STARTS)
    return starts_ok and not starts_bad


def clean_question(text: str) -> str:
    """
    Extract the most likely question line from Gemini's raw response.

    Gemini sometimes adds preamble text or extra lines despite being told not to.
    This function:
      1. Splits the response into non-empty lines.
      2. Scans from the last line backwards to find a line containing '؟'.
      3. Trims anything after the first '؟' (in case of trailing characters).
      4. Falls back to the last line if no '؟' is found anywhere.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return ""
    for line in reversed(lines):
        if "؟" in line:
            return line[:line.index("؟") + 1].strip()
    return lines[-1]


# =============================================================================
# JSON PERSISTENCE — efficient append without reloading the whole file
# =============================================================================

def append_to_json(entry: dict):
    """
    Append a single dict entry to the JSON array stored in OUTPUT_FILE.

    WHY NOT json.dump() THE WHOLE FILE EACH TIME?
    For large datasets, re-reading and re-writing thousands of entries on every
    save is slow and risky (data loss if the script crashes mid-write).

    This function instead uses a binary seek trick:
      - Seeks to the position of the closing ']' character at the end of the file.
      - Inserts the new JSON object just before that ']', separated by a comma.
      - Truncates the file at the new end — no full rewrite needed.

    If the file does not exist yet, it creates it with a valid JSON array.
    """
    new_entry_str = json.dumps(entry, ensure_ascii=False, indent=2)

    # --- Case 1: file does not exist yet — create it from scratch ---
    if not OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("[\n" + new_entry_str + "\n]")
        log.info("JSON file created with first entry")
        return

    # --- Case 2: file exists — find the closing ']' and insert before it ---
    with open(OUTPUT_FILE, "rb+") as f:  # open in binary read+write mode
        # Seek to the very end of the file, then walk backwards byte by byte
        # until we find the ']' character that closes the JSON array.
        f.seek(0, 2)           # move to end-of-file
        pos = f.tell() - 1    # start one byte before EOF

        while pos > 0:
            f.seek(pos)
            ch = f.read(1)
            if ch == b"]":    # found the closing bracket
                break
            pos -= 1

        # Check whether the existing array is empty ("[]", "[\n]", etc.)
        # so we know whether to prepend a comma or not.
        f.seek(0)
        content_start = f.read(10).decode("utf-8").strip()
        is_empty_array = content_start in ["[]", "[\n]", "[ ]"]

        # Overwrite from the position of ']' onwards.
        f.seek(pos)
        if is_empty_array:
            # No comma needed before the first real element.
            f.write(("\n" + new_entry_str + "\n]").encode("utf-8"))
        else:
            # Separate the new entry from the previous one with a comma.
            f.write((",\n" + new_entry_str + "\n]").encode("utf-8"))
        f.truncate()  # discard any leftover bytes beyond the new end

    # Quick sanity-check: try to count entries to confirm the file is valid JSON.
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
    """
    Copy `text` to the system clipboard using the correct tool for each OS.

    WHY NOT use send_keys() to type the text directly?
    - Arabic (right-to-left) text typed character-by-character via Selenium's
      send_keys() is often scrambled or misordered by the OS input pipeline.
    - Pasting from the clipboard preserves the correct Unicode string as-is.

    OS-specific clipboard tools:
      - Windows : PowerShell's Set-Clipboard cmdlet
      - macOS   : pbcopy (built-in, no installation needed)
      - Linux   : xclip (install with: sudo apt install xclip)
    """
    current_os = platform.system()

    if current_os == "Windows":
        # Escape single quotes so they don't break the PowerShell command string.
        escaped = text.replace("'", "''")
        cmd = f"Set-Clipboard -Value '{escaped}'"
        subprocess.run(["powershell", "-command", cmd], check=True, capture_output=True)

    elif current_os == "Darwin":  # Darwin is the internal name for macOS
        # pbcopy reads from stdin and puts the content into the clipboard.
        subprocess.run("pbcopy", input=text.encode("utf-8"), check=True)

    elif current_os == "Linux":
        # xclip reads from stdin; -selection clipboard targets the main clipboard.
        # Must be installed first: sudo apt install xclip
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.encode("utf-8"),
            check=True
        )

    else:
        # Unknown OS — try PowerShell as a last resort.
        log.warning(f"Unknown OS '{current_os}' — trying PowerShell as clipboard fallback")
        escaped = text.replace("'", "''")
        subprocess.run(
            ["powershell", "-command", f"Set-Clipboard -Value '{escaped}'"],
            check=True,
            capture_output=True
        )


# =============================================================================
# HUMAN BEHAVIOUR SIMULATION
# =============================================================================

def human_pause(min_s: float, max_s: float):
    """
    Sleep for a random duration between `min_s` and `max_s` seconds.

    Identical inter-request timing is a strong bot signal. Randomising pauses
    makes the traffic pattern look more like a human browsing session.
    """
    time.sleep(random.uniform(min_s, max_s))


def human_scroll(driver):
    """
    Scroll the page down by a random amount (80-350 px), then pause briefly.

    Real users scroll while reading. Occasional scrolling avoids patterns
    where every request starts from exactly the same viewport position.
    execute_script() runs arbitrary JavaScript inside the browser.
    """
    scroll_px = random.randint(80, 350)
    driver.execute_script(f"window.scrollBy(0, {scroll_px})")
    human_pause(0.3, 0.9)


def check_for_captcha(driver) -> bool:
    """
    Return True if the current page appears to contain a CAPTCHA challenge.

    Searches the full page HTML source for common CAPTCHA-related keywords in
    multiple languages. If a CAPTCHA is detected, the main loop will pause and
    wait for the user to solve it manually.
    """
    indicators = [
        "recaptcha", "captcha", "unusual traffic",
        "verify you're human", "تحقق", "Je ne suis pas un robot"
    ]
    page_text = driver.page_source.lower()
    return any(ind.lower() in page_text for ind in indicators)


def check_for_rate_limit(driver) -> bool:
    """
    Return True if Gemini is showing a rate-limit or quota-exceeded error.

    When too many requests are sent in a short time, Google temporarily blocks
    further requests. Detecting this early lets the script wait and retry instead
    of wasting requests or getting permanently blocked.
    """
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
    """
    Create and return a configured Selenium Chrome WebDriver instance.

    Automatically detects the OS and applies the matching user-agent string.
    platform.system() returns:
      "Windows" — Windows 10 or 11
      "Darwin"  — macOS (any version)
      "Linux"   — Ubuntu, Debian, Fedora, etc.

    Chrome is launched with several settings that reduce its automation footprint:

    --no-sandbox
        Disables Chrome's security sandbox. Required on Linux/CI environments.
        Safe to keep on Windows and macOS — simply ignored there.

    --disable-dev-shm-usage
        Prevents Chrome from using /dev/shm (shared memory) which can cause
        crashes on Linux in memory-constrained environments.
        Harmless on Windows and macOS.

    --disable-blink-features=AutomationControlled
        Removes the "Chrome is being controlled by automated software" banner
        and the corresponding JavaScript property (navigator.webdriver).

    excludeSwitches / useAutomationExtension
        Additional switches to strip further automation metadata from the browser.

    After creation, two low-level tricks are applied via CDP (Chrome DevTools Protocol):
      - navigator.webdriver is overridden to undefined (hides Selenium).
      - The user-agent is enforced at the network layer for full consistency.
    """
    options = Options()

    # Required on Linux; harmless on Windows and macOS.
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # Hide automation indicators from the browser itself.
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    # Random window dimensions to vary the browser fingerprint across runs.
    w = random.choice([1280, 1366, 1440, 1536, 1920])
    h = random.choice([768, 800, 864, 900, 1080])
    options.add_argument(f"--window-size={w},{h}")

    # -------------------------------------------------------------------------
    # AUTO-DETECT OS and select the matching user-agent string.
    # The user-agent tells websites what browser and OS you are using.
    # It must match the real OS to avoid inconsistency detection by Google.
    # -------------------------------------------------------------------------
    current_os = platform.system()
    log.info(f"Detected OS: {current_os}")

    if current_os == "Windows":
        # Windows 10 and Windows 11 share the same identifier (Windows NT 10.0).
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    elif current_os == "Darwin":  # Darwin is the internal kernel name for macOS
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
        # Unknown OS — fall back to Windows user-agent as a safe default.
        log.warning(f"Unknown OS '{current_os}' — falling back to Windows user-agent")
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )

    # Apply the user-agent to the Chrome options.
    options.add_argument(f"--user-agent={user_agent}")

    # If a Chrome profile path is given, use it so the browser is pre-authenticated.
    if profile_path:
        options.add_argument(f"--user-data-dir={profile_path}")
        options.add_argument(f"--profile-directory={profile_dir}")

    # Launch the Chrome browser with all the options configured above.
    driver = webdriver.Chrome(options=options)

    # Hide the navigator.webdriver JavaScript property that websites use to
    # detect Selenium. execute_script() runs JS directly inside the browser tab.
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )

    # Override the user-agent at the Chrome DevTools Protocol (CDP) level as well.
    # This ensures the network-layer user-agent matches the one set in options above.
    # We reuse the same `user_agent` variable — so they are always identical.
    driver.execute_cdp_cmd("Network.setUserAgentOverride", {"userAgent": user_agent})

    return driver


# =============================================================================
# WAITING FOR GEMINI'S RESPONSE
# =============================================================================

def wait_for_response(driver: webdriver.Chrome, timeout: int = 90) -> str:
    """
    Wait until Gemini has finished generating its response and return the text.

    STRATEGY:
      1. Poll for the "Stop generating" button. While it is present, Gemini is
         still streaming its answer — keep waiting.
      2. Once that button disappears (generation is complete), try a series of
         CSS selectors that may contain the response text.
      3. Among all matching elements, take the LAST one — it corresponds to
         the most recent message in the chat.
      4. Fall back to broader selectors ([role='article'], div[class*='response'])
         if the specific ones return nothing.
      5. Return an empty string if nothing is found (the caller will retry).

    WHY MULTIPLE SELECTORS?
      Gemini's DOM structure can change with UI updates. Using a ranked list of
      selectors makes the script more resilient to minor HTML changes.
    """
    human_pause(2, 4)  # brief initial wait before we start polling
    end_time = time.time() + timeout

    # --- Phase 1: wait for generation to finish ---
    while time.time() < end_time:
        try:
            # If this element is found, Gemini is still generating — wait more.
            driver.find_element(By.CSS_SELECTOR, "[aria-label='Stop generating']")
            time.sleep(1)
        except NoSuchElementException:
            # Element not found means generation is complete.
            human_pause(1, 2)
            break

    # --- Phase 2: extract the response text using known CSS selectors ---
    selectors = [
        "model-response .response-content",  # primary Gemini response container
        ".model-response-text",
        "message-content",
        "[data-response-index]",
        ".markdown",
    ]

    for sel in selectors:
        try:
            # find_elements() returns a list (empty list if nothing matches — no exception).
            elements = driver.find_elements(By.CSS_SELECTOR, sel)
            if elements:
                # Take the last element: it is the most recent reply in the chat history.
                text = elements[-1].text.strip()
                if text:
                    return text
        except Exception:
            continue  # selector failed for any reason — try the next one

    # --- Phase 3: fallback selectors ---
    try:
        blocks = driver.find_elements(
            By.CSS_SELECTOR, "[role='article'], div[class*='response']"
        )
        if blocks:
            return blocks[-1].text.strip()
    except Exception:
        pass

    return ""  # nothing found — caller will treat this as a failed attempt


# =============================================================================
# SENDING THE PROMPT TO GEMINI
# =============================================================================

def send_prompt(driver: webdriver.Chrome, prompt: str) -> str:
    """
    Type the prompt into Gemini's input box and return the model's response text.

    STEP-BY-STEP:
      1. Copy the prompt to the clipboard (OS-aware — preserves Arabic text).
      2. Locate the chat input box using a list of CSS selectors.
      3. Click it, select-all, delete existing text (clears any leftovers).
      4. Paste from clipboard with Ctrl+V.
      5. Press Enter to submit.
      6. Call wait_for_response() and return whatever text Gemini produces.

    FINDING THE INPUT BOX:
      WebDriverWait(driver, 15) blocks for up to 15 seconds until the element
      satisfying EC.element_to_be_clickable() exists and is interactable.
      If it times out (TimeoutException), we try the next selector in the list.
    """
    # Step 1 — put the Arabic prompt on the clipboard (method depends on OS)
    copy_to_clipboard(prompt)
    human_pause(0.4, 0.8)

    # Candidate selectors for Gemini's text input area (ordered by specificity).
    input_selectors = [
        "div[contenteditable='true'][role='textbox']",  # rich text editor
        "rich-textarea div[contenteditable]",            # web-component variant
        "textarea",                                      # plain textarea fallback
    ]

    input_box = None
    for sel in input_selectors:
        try:
            # Wait up to 15 s for this element to become clickable, then stop waiting.
            input_box = WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
            )
            break  # found a usable input box — exit the loop
        except TimeoutException:
            continue  # timed out on this selector — try the next one

    if not input_box:
        raise Exception("Gemini text input box not found — page may have changed")

    # Step 2 — click the box to focus it
    input_box.click()
    human_pause(0.2, 0.5)

    # Step 3 — clear any pre-existing text (Ctrl+A then Delete)
    input_box.send_keys(Keys.CONTROL + "a")
    human_pause(0.1, 0.3)
    input_box.send_keys(Keys.DELETE)
    human_pause(0.2, 0.4)

    # Step 4 — paste the Arabic prompt from clipboard (Ctrl+V)
    input_box.send_keys(Keys.CONTROL + "v")
    human_pause(0.8, 1.5)

    # Step 5 — submit with Enter
    input_box.send_keys(Keys.RETURN)
    log.info("Prompt submitted — waiting for response...")

    # Step 6 — wait and return the response
    return wait_for_response(driver)


# =============================================================================
# MAIN — orchestrates the full pipeline
# =============================================================================

def main():
    """
    Entry point. Parses CLI arguments, reads the CSV, drives Chrome, and
    saves results to JSON.

    CLI ARGUMENTS:
      --csv / -c         Path to the CSV file (must have a 'context' column).
      --delay / -d       Minimum seconds to wait between requests (default: 8).
                         Actual delay is randomised to [delay, delay*2].
      --profile / -p     Path to a Chrome user-data directory (for pre-auth).
      --profile-dir      Name of the Chrome profile folder (e.g. "Profile 2").
      --retry            Number of extra attempts if Gemini returns an invalid question.
      --pause-every      Take a long break (90-180 s) every N requests (default: 20).

    RESUME AFTER CRASH:
      Already-processed contexts are tracked in the OUTPUT_FILE JSON.
      On restart, the script loads that file and skips any context it already has,
      so no work is duplicated and nothing is lost.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Darija question generator — cross-platform, anti-ban")
    parser.add_argument("--csv",         "-c", required=True,
                        help="CSV file path (must contain a 'context' column)")
    parser.add_argument("--delay",       "-d", type=int, default=8,
                        help="Minimum delay between requests in seconds (default: 8)")
    parser.add_argument("--profile",     "-p", default=None,
                        help="Chrome user-data-dir path for pre-authenticated profile")
    parser.add_argument("--profile-dir",       default="Default",
                        help="Chrome profile folder name (e.g. 'Profile 2')")
    parser.add_argument("--retry",             type=int, default=2,
                        help="Extra attempts if the generated question is invalid (default: 2)")
    parser.add_argument("--pause-every",       type=int, default=20,
                        help="Take a long break every N requests (default: 20)")
    args = parser.parse_args()

    # ── Read contexts from the CSV file ──────────────────────────────────────
    # pandas reads the CSV into a DataFrame (a table).
    # iloc[START_INDEX:END_INDEX] selects a specific range of rows.
    df = pd.read_csv(args.csv, encoding="utf-8")
    df_slice = df.iloc[START_INDEX:END_INDEX]

    log.info(f"Processing rows {START_INDEX} to {END_INDEX - 1} "
             f"({len(df_slice)} contexts total)")

    # ── Load already-processed contexts to allow resuming after a crash ───────
    # If the output file already exists, load it and collect all context strings
    # that were already processed. These will be skipped in the main loop.
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_tasks = json.load(f)
        log.info(f"Loaded {len(existing_tasks)} already-processed entries from {OUTPUT_FILE}")
    else:
        existing_tasks = []

    # Build a set of context strings we already have — O(1) lookup per context.
    processed_texts = set(task["data"]["context"] for task in existing_tasks)

    # ── Launch Chrome and navigate to Gemini ─────────────────────────────────
    driver = create_driver(args.profile, args.profile_dir)
    driver.get(GEMINI_URL)   # open the Gemini chat page in the browser
    human_pause(4, 7)        # wait for the page to fully load

    # If Chrome is not authenticated, Google redirects to the sign-in page.
    # Pause here and let the user log in manually, then continue.
    if "accounts.google.com" in driver.current_url or "signin" in driver.current_url:
        log.warning("Not signed in — please log in to Google in the browser window...")
        input("Press Enter after signing in: ")

    log.info("Connected to Gemini — starting processing loop")

    # ── Main processing loop ──────────────────────────────────────────────────
    success      = 0   # count of successfully saved (context, question) pairs
    skipped      = 0   # count of contexts where all retry attempts failed
    already_done = 0   # count of contexts skipped because already processed

    # iterrows() yields (index, row) pairs where row is a pandas Series.
    # We use enumerate to track position within the current slice (for pause logic).
    for loop_pos, (csv_index, row) in enumerate(df_slice.iterrows(), start=1):

        # Extract the plain text string from the 'context' column of this row.
        # row is a pandas Series — row["context"] gives us the actual string value.
        context_text = str(row["context"]).strip()

        # Skip this context if it was already processed in a previous run.
        if context_text in processed_texts:
            already_done += 1
            log.info(f"[{loop_pos}] Row #{csv_index} already processed — skipping")
            continue

        log.info(f"[{loop_pos}] Processing row #{csv_index}...")

        # --- Safety checks before sending the next request ---

        # If a CAPTCHA appeared, pause and wait for manual resolution.
        if check_for_captcha(driver):
            log.warning("CAPTCHA detected — please solve it manually in the browser...")
            input("Press Enter after solving the CAPTCHA: ")

        # If we hit a rate limit, wait a random time then refresh the page.
        if check_for_rate_limit(driver):
            wait_time = random.randint(60, 120)
            log.warning(f"Rate limit detected — waiting {wait_time}s before retrying...")
            time.sleep(wait_time)
            driver.refresh()    # reload the page to clear the error state
            human_pause(3, 6)

        # Every `pause_every` requests, take a long break to cool down.
        if loop_pos > 1 and loop_pos % args.pause_every == 0:
            pause = random.randint(90, 180)
            log.info(f"Long pause ({pause}s) after {loop_pos} requests...")
            time.sleep(pause)
            human_scroll(driver)   # scroll a bit to look active after the pause

        # Scroll before each request to simulate natural reading behaviour.
        human_scroll(driver)

        # --- Generate the question, with retries for invalid responses ---
        question = None

        # Total attempts = 1 initial + args.retry extra = args.retry + 1 total.
        for attempt in range(1, args.retry + 2):
            try:
                # Fill the prompt template with the current context and send it.
                raw = send_prompt(driver, PROMPT_TEMPLATE.format(context=context_text))
            except Exception as e:
                log.error(f"Error sending prompt: {e}")
                break   # unrecoverable error for this context — give up

            # Clean the raw response to isolate the question line.
            candidate = clean_question(raw)

            if is_valid_question(candidate):
                question = candidate
                log.info(f"Valid question: {question}")
                break
            else:
                log.warning(
                    f"Attempt {attempt} invalid: '{candidate[:60]}' — retrying..."
                )
                human_pause(3, 6)   # brief pause before the next attempt

        # --- Persist the result immediately ---
        if question:
            entry = {
                "data": {
                    "context" : context_text,
                    "question": question,
                }
            }
            append_to_json(entry)
            success += 1
        else:
            log.error(
                f"Row #{csv_index} skipped after {args.retry + 1} failed attempts"
            )
            skipped += 1

        # Random delay before the next request.
        delay = random.uniform(args.delay, args.delay * 2)
        log.info(f"Waiting {delay:.1f}s before next request...")
        time.sleep(delay)

    # ── Final summary ─────────────────────────────────────────────────────────
    driver.quit()   # close the Chrome browser
    log.info(f"Done: {success} generated, {skipped} failed, {already_done} already done")
    log.info(f"Output file: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()