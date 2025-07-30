# ml_service/submission_handler/document_loader.py

import os
import uuid
import logging
import requests
import time
import asyncio
import magic  # python-magic for file type detection
from typing import Tuple, Optional
from urllib.parse import urlparse
import re

# Constants for supported file types
SUPPORTED_MIMETYPES = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/msword': '.doc',
    'text/plain': '.txt'
}

class UnsupportedFileTypeError(Exception):
    """Raised when file type is not supported"""
    pass

def _detect_file_type(content: bytes) -> str:
    """Detects file type from content using python-magic."""
    mime = magic.Magic(mime=True)
    return mime.from_buffer(content)

def _validate_url(url: str) -> bool:
    """Validates if the URL is properly formatted."""
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except Exception:
        return False

def _sanitize_filename(filename: str) -> str:
    """Removes characters that could be problematic in a filename."""
    return re.sub(r'[^a-zA-Z0-9._-]', '', filename)


def download_file(url: str, dest_folder: str, chunk_size: int = 8192) -> Tuple[str, str]:
    """
    Downloads a file from a URL with streaming, validation, and robust error handling.
    """
    if not _validate_url(url):
        raise ValueError(f"Invalid or unsupported URL: {url}")

    os.makedirs(dest_folder, exist_ok=True)
    start_time = time.time()

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').split(';')[0]
        
        # --- FIX for File Corruption ---
        # Read the first chunk to detect the type if necessary, then write it.
        first_chunk = next(response.iter_content(chunk_size=1024), None)
        if not first_chunk:
            raise ValueError("Downloaded file is empty.")

        if content_type not in SUPPORTED_MIMETYPES:
            logging.info(f"Content-Type header '{content_type}' not found or unsupported. Detecting from content.")
            content_type = _detect_file_type(first_chunk)
        # --- END OF FIX ---

        if content_type not in SUPPORTED_MIMETYPES:
            raise UnsupportedFileTypeError(f"Unsupported file type detected: {content_type}")
        
        ext = SUPPORTED_MIMETYPES[content_type]
        orig_filename = _sanitize_filename(url.split("/")[-1].split("?")[0])
        filename = orig_filename if orig_filename.endswith(ext) else f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(dest_folder, filename)
        
        total_size = 0
        with open(save_path, 'wb') as f:
            # Write the first chunk that we already read
            f.write(first_chunk)
            total_size += len(first_chunk)
            
            # Write the rest of the file
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        elapsed = time.time() - start_time
        logging.info(f"Downloaded '{filename}' ({total_size / 1024:.2f} KB) to '{save_path}' in {elapsed:.2f}s")
        return save_path, filename


async def async_download_file(url: str, dest_folder: str) -> Optional[Tuple[str, str]]:
    """
    Asynchronous wrapper for download_file that handles exceptions.
    """
    try:
        return await asyncio.to_thread(download_file, url, dest_folder)
    except Exception as e:
        # The main handler will catch and log this.
        # We log it here as well for more specific context.
        logging.error(f"File download failed for URL '{url}': {e}")
        return None, None
