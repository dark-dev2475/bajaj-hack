# ml_service/submission_handler/downloader.py

import os
import uuid
import logging
import requests
import time
import asyncio
from typing import Tuple


def download_file(url: str, dest_folder: str) -> Tuple[str, str]:
    """
    Synchronous version - Downloads a file from the given URL.
    """
    os.makedirs(dest_folder, exist_ok=True)
    start_time = time.time()
    response = requests.get(url, timeout=300)  # 5 minutes timeout
    response.raise_for_status()
    download_time = time.time() - start_time

    filename = url.split("/")[-1].split("?")[0] or f"{uuid.uuid4().hex}.tmp"
    save_path = os.path.join(dest_folder, filename)

    with open(save_path, "wb") as f:
        f.write(response.content)

    logging.info(f"[Timing] Downloaded '{filename}' ({len(response.content)} bytes) to '{save_path}' in {download_time:.2f}s")
    return save_path, filename


async def async_download_file(url: str, dest_folder: str) -> Tuple[str, str]:
    """
    Asynchronous wrapper for download_file using asyncio.to_thread.
    Keeps same logic and variable names.
    """
    return await asyncio.to_thread(download_file, url, dest_folder)
