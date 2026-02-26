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
# CHANGE: Sync to Async
class ExternalContentCache:
    def __init__(self):
        self.blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
        if not self.blob_connection_string:
            raise ValueError("Missing AZURE_BLOB_CONNECTION_STRING environment variable")
 
        self.container_name = os.getenv("AZURE_BLOB_CONTAINER")
        # Use the async client by creating it within an async context or at startup
        self.blob_service_client = BlobServiceClient.from_connection_string(self.blob_connection_string)
        self.scraped_folder = "scraped_external_content"
    
    def _blob_name(self, url: str) -> str:
        # sanitize url to use as blob name
        sanitized = url.replace("://", "_").replace("/", "_")
        return f"externalcache/{self.scraped_folder}/{sanitized}.md"
    
    async def get_cached_markdown(self, url: str) -> tuple[str, Dict[str, Any]]:
        # Retrieve cached markdown content from Azure Blob Storage if exists, else return None
        blob_name = self._blob_name(url)
        try:
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob_name)
            download_stream = await blob_client.download_blob()
            data = await download_stream.readall()
            content = data.decode('utf-8')
        
            # Get blob properties for cache metadata
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


# Global cache client instance (lazy initialization)
_cache_client_instance = None

def get_cache_client() -> ExternalContentCache:
    """Get or create the singleton cache client instance"""
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

    """
    Main scraping function:
    - check cache unless force_rebuild is True
    - Use Scraping Robot API to fetch rendered HTML output of the URL
      (handles proxies, retries, JS rendering transparently)
    - extract main content using readability-lxml
    - convert content to markdown via html2text
    - cache the markdown to Azure Blob Storage
    - append citation info to markdown
    - return markdown string
    """
    if not force_rebuild:
        cached, cache_meta = await get_cache_client().get_cached_markdown(url)
        if cached:
            metadata.cache_hit = True
            metadata.content_length = cache_meta.get("content_length", len(cached))
            metadata.scraped_at = cache_meta.get("cached_at")
            metadata.fetch_time_ms = (time.time() - start_time) * 1000

            logger.info(f"Cache HIT for external URL: {url} (fetch time: {metadata.fetch_time_ms:.2f}ms)")
            
            if include_metadata:
                return {
                    "content": cached,
                    "metadata": metadata.to_dict()
                }

            return {"content": cached}
        
    logger.info(f"Cache MISS for external URL: {url} - scraping...")
    
    # Retrieve Scraping Robot API key from environment variable
    scrapingrobot_api_key = os.getenv("SCRAPINGROBOT_API_KEY")
    if not scrapingrobot_api_key:
        raise ValueError("Missing SCRAPINGROBOT_API_KEY environment variable")

    api_endpoint = "https://api.scrapingrobot.com"
    params = {
        "token": scrapingrobot_api_key,
        "url": url,
        # "render": "true"  # Optionally add if JavaScript rendering is needed
    }
    
    html = ""
    fetch_start = time.time()

    try:
        # Make an async GET request to Scraping Robot API to fetch the fully rendered HTML
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(api_endpoint, params=params)
            resp.raise_for_status()

            logger.info(f"📡 API Response Status: {resp.status_code}")
            logger.info(f"📡 API Response Preview: {resp.text[:500]}")

            response_data = resp.json()

            logger.info(f"📦 Full response data: {json.dumps(response_data, indent=2)[:1000]}")

            # Check if the API call was successful
            if response_data.get("status") == "SUCCESS":
                # Extract the HTML from the 'result' field
                html = response_data.get("result", "")
                logger.info(f"Fetched content from Scraping Robot API for URL: {url}")

                logger.info(f"Raw HTML length: {len(html)} characters")
                logger.info(f"HTML preview (first 2000 chars):\n{html[:2000]}")
                logger.info(f"Does HTML contain 'Python is a': {'Python is a' in html}")
                logger.info(f"Full response data keys: {response_data.keys()}")

            else:
                error_msg = f"Scraping Robot API returned status: {response_data.get('status')}"
                logger.error(error_msg)
                metadata.fetch_time_ms = (time.time() - start_time) * 1000
                return {
                    "content": f"Error: {error_msg}",
                    "metadata": metadata.to_dict() if include_metadata else None
                }

    except httpx.RequestError as e:
        logging.error(f"Failed to fetch URL '{url}' via Scraping Robot API: {e}")
        metadata.fetch_time_ms = (time.time() - start_time) * 1000
        return {
            "content": f"Error fetching content from {url}: {e}",
            "metadata": metadata.to_dict() if include_metadata else None
        }
    
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON response from Scraping Robot API: {e}")
        metadata.fetch_time_ms = (time.time() - start_time) * 1000
        return {
            "content": f"Error: Invalid JSON response from scraping service",
            "metadata": metadata.to_dict() if include_metadata else None
        }

    # Extract main content from fetched HTML using readability-lxml
    # Run blocking HTML extraction in thread to avoid blocking event loop
    try:
        def parse_html_content(html):
            doc = Document(html)
            return doc.summary()
        
        main_html = await asyncio.to_thread(parse_html_content, html)
        logger.info(f"Extracted main content for URL: {url}")
    except Exception as e:
        logging.error(f"Failed to extract main content from HTML for URL '{url}': {e}")
        main_html = html # fallback to full html if extraction fails
    
    # Run blocking markdown conversion in thread
    try:
        def convert_to_markdown(html_content):
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            return h.handle(html_content)
        
        markdown_content = await asyncio.to_thread(convert_to_markdown, main_html)
        logger.info(f"Converted content to markdown for URL: {url}")
    except Exception as e:
        logging.error(f"Failed to convert HTML to Markdown for URL '{url}': {e}")
        markdown_content = "WARNING: Content conversion failed."
    
    # Append citation for transparency
    citation = f"\n\n---\n\n*Content sourced from {url}*"
    full_markdown = markdown_content + citation

    metadata.content_length = len(full_markdown)
    metadata.fetch_time_ms = (time.time() - start_time) * 1000
    metadata.scraped_at = datetime.now().isoformat()
    
    # Cache results
    await get_cache_client().save_markdown(url, full_markdown)
    
    logger.info(f"Scrape complete for {url} (fetch time: {metadata.fetch_time_ms:.2f}ms)")
    
    if include_metadata:
        return {
            "content": full_markdown,
            "metadata": metadata.to_dict()
        }
    return {"content": full_markdown}