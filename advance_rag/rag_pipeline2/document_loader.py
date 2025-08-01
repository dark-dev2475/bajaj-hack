# /rag_pipeline/document_loader.py

import os
import requests
import logging
from typing import List, Dict
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoader:
    """Handles downloading documents from URLs."""
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    async def download_files(self, urls: List[str]) -> List[Dict]:
        """
        Asynchronously downloads files from a list of URLs.
        (Using sync requests here for simplicity in this context)
        """
        results = []
        for url in urls:
            try:
                filename = os.path.basename(urlparse(url).path)
                file_path = os.path.join(self.data_dir, filename)

                response = requests.get(url, stream=True, timeout=300)
                response.raise_for_status()

                with open(file_path, "wb") as f:
                    f.write(response.content)
                
                logger.info(f"Successfully downloaded {url} to {file_path}")
                results.append({"url": url, "file_path": file_path, "status": "success"})
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download {url}. Error: {e}")
                results.append({"url": url, "file_path": None, "status": "error", "reason": str(e)})
        return results