import os
import time
import logging
from openai import OpenAI
from typing import Union, List

# ----------------------------
# Setup logging
# ----------------------------
logger = logging.getLogger("AI_Client")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

# ----------------------------
# Initialize OpenAI client
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set! AI features will not work.")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ----------------------------
# Main AI analysis function
# ----------------------------
def ai_analysis_text(prompt: Union[str, List[str]], retries: int = 3, delay: float = 2.0) -> str:
    """
    Generate AI-based analysis text using OpenAI API.
    
    Args:
        prompt: A single prompt string or a list of prompts.
        retries: Number of retries if API fails.
        delay: Delay between retries (seconds).
    
    Returns:
        AI response text or error message.
    """
    if not client:
        return "⚠️ AI client not initialized. Set OPENAI_API_KEY."

    if isinstance(prompt, list):
        prompt_text = "\n".join(prompt)
    else:
        prompt_text = prompt

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a crypto trading assistant. Provide clear, concise, actionable market analysis."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.7,
                max_tokens=400
            )

            text = response.choices[0].message.content.strip()
            logger.info("AI response successfully generated.")
            return text

        except Exception as e:
            logger.error(f"[AI ERROR] Attempt {attempt}/{retries}: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                return "⚠️ AI service is currently unavailable. Please try again later."

# ----------------------------
# Helper: batch processing for multiple pairs
# ----------------------------
def ai_market_brief_for_pairs(pairs: List[str]) -> str:
    """
    Build a concise AI market brief for multiple trading pairs.
    """
    if not pairs:
        return "⚠️ No trading pairs provided."

    prompt = "Provide a concise 5-sentence market analysis and a one-line BUY/SELL/HOLD verdict for each of the following crypto pairs:\n"
    for p in pairs:
        prompt += f"- {p}\n"

    return ai_analysis_text(prompt)
