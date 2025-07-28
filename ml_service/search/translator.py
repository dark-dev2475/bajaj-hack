import logging
from langdetect import detect
from clients import openai_async_client


# ASYNC translator
async def translate_to_english_async(text: str, openai_client:openai_async_client) -> str:
    try:
        if detect(text) != "en":
            logging.info("Translating query to English (async)...")
            trans_resp = await openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": f"Translate to English: \"{text}\""}]
            )
            return trans_resp.choices[0].message.content.strip()
        return text
    except Exception as e:
        logging.error(f"[Async Translator] Translation failed: {e}")
        raise RuntimeError(f"Translation failed: {e}")
