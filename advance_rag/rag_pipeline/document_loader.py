import os
import logging
import mimetypes
import tempfile
from pathlib import Path
from typing import Optional, Union, List
import aiohttp
import asyncio
from urllib.parse import urlparse, unquote
from pinecone import Pinecone, ServerlessSpec
# Try to import magic, but handle gracefully if not available
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    logging.warning("python-magic not available. Will use mimetypes for file detection.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentLoader:
    """A class to handle downloading and loading documents from various sources."""
    
    SUPPORTED_MIME_TYPES = {
        # PDF files
        'application/pdf': '.pdf',
        
        # Word documents
        'application/msword': '.doc',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        
        # Text files
        'text/plain': '.txt',
        'text/csv': '.csv',
        
        # Email files
        'message/rfc822': '.eml',
        'application/vnd.ms-outlook': '.msg',
        
        # HTML files
        'text/html': '.html',
        
        # PowerPoint
        'application/vnd.ms-powerpoint': '.ppt',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
        
        # Excel
        'application/vnd.ms-excel': '.xls',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx'
    }
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the DocumentLoader.
        
        Args:
            data_dir: Directory where downloaded files will be stored
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    async def _download_file(self, url: str, session: aiohttp.ClientSession) -> Optional[tuple[Path, str]]:
        """
        Download a file from a URL.
        
        Args:
            url: URL of the file to download
            session: aiohttp session to use for download
            
        Returns:
            Tuple of (file path, detected mime type) if successful, None otherwise
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Failed to download {url}. Status: {response.status}")
                    return None
                
                # Get content type and filename
                content_type = response.headers.get('content-type', '').split(';')[0]
                
                # Try to get filename from Content-Disposition header
                cd = response.headers.get('content-disposition')
                filename = None
                if cd:
                    import re
                    if (matches := re.findall("filename=(.+)", cd)):
                        filename = matches[0].strip('"')
                
                # If no filename in headers, get it from URL
                if not filename:
                    filename = unquote(os.path.basename(urlparse(url).path))
                
                # If still no filename, create one based on content type
                if not filename:
                    ext = self.SUPPORTED_MIME_TYPES.get(content_type, '.bin')
                    filename = f"document_{hash(url)}{ext}"
                
                file_path = self.data_dir / filename
                
                # Download the file
                content = await response.read()
                
                # Use python-magic to detect file type if available, otherwise use mimetypes
                if MAGIC_AVAILABLE:
                    mime_type = magic.from_buffer(content, mime=True)
                else:
                    # Fallback to mimetypes based on filename
                    mime_type, _ = mimetypes.guess_type(filename)
                    if not mime_type:
                        mime_type = content_type or 'application/octet-stream'
                
                if mime_type not in self.SUPPORTED_MIME_TYPES:
                    logger.warning(f"Unsupported mime type {mime_type} for {url}")
                    return None
                
                # Save the file
                with open(file_path, 'wb') as f:
                    f.write(content)
                
                logger.info(f"Successfully downloaded {url} to {file_path}")
                return file_path, mime_type
                
        except Exception as e:
            logger.exception(f"Error downloading {url}: {str(e)}")
            return None
    
    async def download_files(self, urls: Union[str, List[str]]) -> List[dict]:
        """
        Download files from one or more URLs.
        
        Args:
            urls: Single URL or list of URLs to download
            
        Returns:
            List of dictionaries containing file info
        """
        if isinstance(urls, str):
            urls = [urls]
            
        results = []
        async with aiohttp.ClientSession() as session:
            tasks = [self._download_file(url, session) for url in urls]
            downloaded = await asyncio.gather(*tasks)
            
            for url, result in zip(urls, downloaded):
                if result:
                    file_path, mime_type = result
                    results.append({
                        "url": url,
                        "file_path": str(file_path),
                        "mime_type": mime_type,
                        "status": "success"
                    })
                else:
                    results.append({
                        "url": url,
                        "file_path": None,
                        "mime_type": None,
                        "status": "failed"
                    })
        
        return results