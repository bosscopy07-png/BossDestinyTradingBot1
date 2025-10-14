import os
from openai import OpenAI

# Load your API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Initialize client (no 'proxies' argument — fully compatible with latest OpenAI)
client = OpenAI(api_key=OPENAI_API_KEY)

def ai_analysis_text(prompt: str) -> str:
    """
    Generate AI-based analysis text using OpenAI API.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # fast and smart model
            messages=[
                {"role": "system", "content": "You are a crypto trading assistant that provides clear, concise market analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )

        # ✅ Return the AI response text
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[AI ERROR] {e}")
        return "⚠️ AI service is currently unavailable. Please try again later."
