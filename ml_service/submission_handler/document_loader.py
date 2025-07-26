# ml_service/submission_handler/downloader.py

import os
import uuid
import logging
import requests
import time
from typing import Tuple

def download_file(url: str, dest_folder: str) -> Tuple[str, str]:
    """
    Downloads a file from the given URL into dest_folder, logging time taken.
    Returns:
        save_path (str): Full filesystem path to the downloaded file.
        filename (str): The filename used.
    Raises:
        HTTPError if the download fails.
    """
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)

    # Start timing
    start_time = time.time()

    # Perform the HTTP GET
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    # Compute download duration
    download_time = time.time() - start_time

    # Determine filename (fall back to random UUID)
    filename = url.split("/")[-1].split("?")[0] or f"{uuid.uuid4().hex}.tmp"
    save_path = os.path.join(dest_folder, filename)

    # Write file to disk
    with open(save_path, "wb") as f:
        f.write(response.content)

    logging.info(f"[Timing] Downloaded '{filename}' ({len(response.content)} bytes) to '{save_path}' in {download_time:.2f}s")
    return save_path, filename
