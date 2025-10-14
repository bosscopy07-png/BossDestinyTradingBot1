# ai_client.py
import os
import traceback
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

def ai_analysis_text(prompt):
    if not client:
        return "AI unavailable (OPENAI_API_KEY not set)."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional crypto market analyst. Provide concise trade rationale, risk controls, and a one-line BUY/SELL verdict."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350,
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        traceback.print_exc()
        return f"AI error: {e}"
