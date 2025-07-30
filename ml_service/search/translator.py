# search/translator.py

import logging
from langdetect import detect
from langchain_core.language_models.chat_models import BaseChatModel

async def translate_to_english_async(text: str, llm: BaseChatModel) -> str:
    """
    Translates text to English if it's not already in English, using the provided LLM.
    
    Args:
        text: The text to translate.
        llm: The OpenAI language model to use for translation.

    Returns:
        The translated English text.
    """
    try:
        # Use a simple check to avoid unnecessary API calls for English text
        if detect(text) == "en":
            return text
            
        logging.info(f"Translating text to English: '{text[:50]}...'")
        
        # Use the invoke method on the LLM, which is the standard for LangChain models
        response = await llm.ainvoke(f"Translate the following text to English: \"{text}\"")
        
        # The response from a LangChain model is an AIMessage object with a 'content' attribute
        translated_text = response.content.strip()
        
        logging.info(f"Translation successful: '{translated_text[:50]}...'")
        return translated_text

    except Exception as e:
        logging.error(f"Translation failed for text '{text[:50]}...': {e}. Returning original text.")
        # In case of failure, it's safer to return the original text than to stop the process.
        return text
