import os
import re
import httpx
import logging
import math
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

def is_youtube_url(url: str) -> bool:
    """
    Checks if the URL is a valid YouTube video URL.
    """
    # Regex covers standard youtube.com, youtu.be, and shorts
    youtube_regex = (
        r'(https?://)?(www\.)?'
        r'(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=|shorts/)?([^&=%\?]{11})'
    )
    return bool(re.match(youtube_regex, url))

def _get_video_id(url: str) -> str:
    """Extracts the 11-character video ID from the URL."""
    youtube_regex = (
        r'(https?://)?(www\.)?'
        r'(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=|shorts/)?([^&=%\?]{11})'
    )
    match = re.match(youtube_regex, url)
    if match:
        return match.group(6)
    return None

def _format_timestamp(seconds: float) -> str:
    """Converts seconds to MM:SS format."""
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

async def fetch_youtube_transcript(url: str) -> str:
    """
    Fetches transcript from Transcript API and converts it to Markdown with clickable timestamps.
    """
    api_key = os.getenv("TRANSCRIPT_API_KEY")
    if not api_key:
        raise ValueError("Missing TRANSCRIPT_API_KEY environment variable")

    # Extract Video ID to build the timestamp links later
    video_id = _get_video_id(url)
    base_video_url = f"https://www.youtube.com/watch?v={video_id}"

    api_url = "https://transcriptapi.com/api/v2/youtube/transcript"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    params = {
        "video_url": url,
        "format": "json", 
        "include_timestamp": "true",
        "send_metadata": "true"
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(api_url, headers=headers, params=params)
        
        if response.status_code == 402:
            raise Exception("Transcript API: Payment Required / Credits Exhausted")
        if response.status_code == 404:
            raise Exception("Transcript API: Video not found or no transcript available")
        
        response.raise_for_status()
        data = response.json()

    # --- Format to Markdown ---
    markdown_output = []
    
    # 1. Add Metadata Header
    metadata = data.get("metadata", {})
    title = metadata.get("title", "Unknown Video")
    author = metadata.get("author_name", "Unknown Channel")
    
    markdown_output.append(f"# Transcript: {title}")
    markdown_output.append(f"**Channel:** {author}")
    markdown_output.append(f"**Source:** {url}")
    markdown_output.append("\n---\n")

    # 2. Add Transcript Segments
    segments = data.get("transcript", [])
    
    if not segments and "text" in data:
        # Fallback if format was text or structure is flat
        markdown_output.append(data["text"])
    else:
        for segment in segments:
            start_time = segment.get("start", 0)
            text = segment.get("text", "").strip()
            
            # Create visual timestamp (e.g., 04:20)
            timestamp_text = _format_timestamp(start_time)
            
            # Create clickable link (e.g., https://youtube.com/watch?v=xyz&t=260s)
            # We cast to int to remove decimals for the URL
            timestamp_link = f"{base_video_url}&t={int(start_time)}s"
            
            # Format: [**[04:20]**](url) The text content here...
            markdown_output.append(f"[**[{timestamp_text}]**]({timestamp_link}) {text}")

    return "\n\n".join(markdown_output)