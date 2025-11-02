# ai_client.py
import os
import time
import logging

logger = logging.getLogger("AI_Client")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = None
try:
    if OPENAI_API_KEY:
        # prefer new OpenAI python library if available
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
        except Exception:
            import openai
            openai.api_key = OPENAI_API_KEY
            client = openai
except Exception:
    client = None

def ai_analysis_text(prompt, retries=2):
    if not client:
        return "⚠️ AI not configured. Set OPENAI_API_KEY to enable AI features."
    for attempt in range(1, retries+1):
        try:
            # try modern interface first
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                resp = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[{"role":"system","content":"You are a crypto trading assistant."},
                              {"role":"user","content":prompt}],
                    temperature=0.6,
                    max_tokens=350
                )
                text = resp.choices[0].message.content.strip()
                return text
            else:
                # fallback to openai.Completion or ChatCompletion
                if hasattr(client, "ChatCompletion"):
                    resp = client.ChatCompletion.create(
                        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                        messages=[{"role":"system","content":"You are a crypto trading assistant."},
                                  {"role":"user","content":prompt}],
                        temperature=0.6,
                        max_tokens=350
                    )
                    return resp.choices[0].message.content.strip()
                else:
                    return "AI client available but interface unsupported."
        except Exception as e:
            logger.error("AI attempt %s failed: %s", attempt, e)
            time.sleep(1)
    return "⚠️ AI service currently unavailable."
