import logging
import httpx
from readability import Document
import html2text
from azure.storage.blob.aio import BlobServiceClient
import os
from dotenv import load_dotenv
import asyncio
from typing import Dict, Any
from dataclasses import dataclass, asdict
import json
import time
from datetime import datetime

# Import the new YouTube handler
from apis.youtube_transcript import is_youtube_url, fetch_youtube_transcript

load_dotenv()

# Module specific log parsing
logger = logging.getLogger(__name__)

@dataclass
class ScraperMetadata:
    """Lightweight metadata for scraper responses"""
    cache_hit: bool
    source_url: str
    fetch_time_ms: float = 0
    content_length: int = 0
    scraped_at: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# Azure Blob Storage setup for scraped content caching
class ExternalContentCache:
    def __init__(self):
        self.blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
        if not self.blob_connection_string:
            raise ValueError("Missing AZURE_BLOB_CONNECTION_STRING environment variable")
 
        self.container_name = os.getenv("AZURE_BLOB_CONTAINER")
        self.blob_service_client = BlobServiceClient.from_connection_string(self.blob_connection_string)
        self.scraped_folder = "scraped_external_content"
    
    def _blob_name(self, url: str) -> str:
        # sanitize url to use as blob name
        sanitized = url.replace("://", "_").replace("/", "_")
        return f"externalcache/{self.scraped_folder}/{sanitized}.md"
    
    async def get_cached_markdown(self, url: str) -> tuple[str, Dict[str, Any]]:
        blob_name = self._blob_name(url)
        try:
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob_name)
            download_stream = await blob_client.download_blob()
            data = await download_stream.readall()
            content = data.decode('utf-8')
        
            properties = await blob_client.get_blob_properties()

            metadata = {
                    "cached_at": properties.last_modified.isoformat() if properties.last_modified else None,
                    "content_length": len(content)
                }
            
            return content, metadata

        except Exception as e:
            logger.info(f"Cache MISS for '{url}': {e}")
            return None, None
    
    async def save_markdown(self, url: str, markdown: str) -> None:
        blob_name = self._blob_name(url)
        try:
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob_name)
            async with blob_client:
                await blob_client.upload_blob(markdown, overwrite=True)
            logger.info(f"Scraped content cached to blob '{blob_name}'.")
        except Exception as e:
            logging.error(f"Failed to save scraped content to blob '{blob_name}': {e}")


_cache_client_instance = None

def get_cache_client() -> ExternalContentCache:
    global _cache_client_instance
    if _cache_client_instance is None:
        _cache_client_instance = ExternalContentCache()
    return _cache_client_instance


async def scrape_external_url_to_markdown(url: str, force_rebuild: bool = False, include_metadata: bool = True) -> Dict[str, Any]:
    start_time = time.time()
    logger.info(f"--- Starting scrape for URL: {url} (force_rebuild={force_rebuild}) ---")
    
    metadata = ScraperMetadata(
        cache_hit=False,
        source_url=url
    )

    # 1. Check Cache
    if not force_rebuild:
        cached, cache_meta = await get_cache_client().get_cached_markdown(url)
        if cached:
            metadata.cache_hit = True
            metadata.content_length = cache_meta.get("content_length", len(cached))
            metadata.scraped_at = cache_meta.get("cached_at")
            metadata.fetch_time_ms = (time.time() - start_time) * 1000

            logger.info(f"Cache HIT for external URL: {url}")
            
            if include_metadata:
                return {
                    "content": cached,
                    "metadata": metadata.to_dict()
                }
            return {"content": cached}
        
    logger.info(f"Cache MISS for external URL: {url} - scraping...")
    
    full_markdown = ""

    # 2. Determine Strategy: YouTube API vs Scraping Robot
    try:
        if is_youtube_url(url):
            logger.info(f"Detected YouTube URL: {url}. Using Transcript API.")
            full_markdown = await fetch_youtube_transcript(url)
        else:
            # --- Existing Scraping Robot Logic ---
            scrapingrobot_api_key = os.getenv("SCRAPINGROBOT_API_KEY")
            if not scrapingrobot_api_key:
                raise ValueError("Missing SCRAPINGROBOT_API_KEY environment variable")

            api_endpoint = "https://api.scrapingrobot.com"
            params = {
                "token": scrapingrobot_api_key,
                "url": url,
            }
            
            html = ""
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(api_endpoint, params=params)
                resp.raise_for_status()
                response_data = resp.json()

                if response_data.get("status") == "SUCCESS":
                    html = response_data.get("result", "")
                else:
                    raise Exception(f"Scraping Robot API status: {response_data.get('status')}")

            # Extract content
            def parse_html_content(html_str):
                doc = Document(html_str)
                return doc.summary()
            
            main_html = await asyncio.to_thread(parse_html_content, html)
            
            # Convert to Markdown
            def convert_to_markdown(html_content):
                h = html2text.HTML2Text()
                h.ignore_links = False
                h.ignore_images = True
                return h.handle(html_content)
            
            markdown_content = await asyncio.to_thread(convert_to_markdown, main_html)
            
            # Append citation
            full_markdown = markdown_content + f"\n\n---\n\n*Content sourced from {url}*"

    except Exception as e:
        logging.error(f"Failed to scrape/fetch URL '{url}': {e}")
        metadata.fetch_time_ms = (time.time() - start_time) * 1000
        return {
            "content": f"Error fetching content: {str(e)}",
            "metadata": metadata.to_dict() if include_metadata else None
        }

    # 3. Save to Cache & Return
    metadata.content_length = len(full_markdown)
    metadata.fetch_time_ms = (time.time() - start_time) * 1000
    metadata.scraped_at = datetime.now().isoformat()
    
    await get_cache_client().save_markdown(url, full_markdown)
    
    if include_metadata:
        return {
            "content": full_markdown,
            "metadata": metadata.to_dict()
        }
    return {"content": full_markdown}