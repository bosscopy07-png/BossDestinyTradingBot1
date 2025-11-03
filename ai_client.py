# ai_client.py
import os
import time
import logging
import traceback

# -----------------------------
# LOGGING SETUP
# -----------------------------
logger = logging.getLogger("AI_Client")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

# -----------------------------
# OPENAI CLIENT INITIALIZATION
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = None
try:
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("[AI] Using modern OpenAI client ✅")
        except Exception:
            import openai
            openai.api_key = OPENAI_API_KEY
            client = openai
            logger.info("[AI] Using legacy OpenAI client ✅")
    else:
        logger.warning("[AI] No API key found — AI features disabled ❌")
except Exception as e:
    logger.error(f"[AI] Initialization failed: {e}")
    client = None

# -----------------------------
# CORE FUNCTION
# -----------------------------
def ai_analysis_text(prompt, retries=3, temperature=0.6, max_tokens=400):
    """
    Generate AI-based text responses.
    Used for market briefs, analysis summaries, and smart insights.
    """
    if not client:
        return "⚠️ AI not configured. Please set OPENAI_API_KEY."

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"[AI] Processing request (attempt {attempt})...")
            # --- Modern interface (preferred) ---
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a professional crypto trading AI analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                text = resp.choices[0].message.content.strip()
                logger.info("[AI] Response generated successfully ✅")
                return text

            # --- Legacy ChatCompletion ---
            elif hasattr(client, "ChatCompletion"):
                resp = client.ChatCompletion.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a professional crypto trading AI analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                text = resp.choices[0].message.content.strip()
                logger.info("[AI] Response generated via legacy interface ✅")
                return text

            else:
                return "⚠️ AI interface unsupported by current library version."

        except Exception as e:
            logger.error(f"[AI] Attempt {attempt} failed: {e}")
            traceback.print_exc()
            time.sleep(2)

    return "⚠️ AI service currently unavailable. Try again later."
