import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from apis.external_content_scraper import scrape_external_url_to_markdown

router = APIRouter()

class ScrapeRequest(BaseModel):
    url: str
    force_rebuild: bool = False
    include_metadata: bool = True

@router.post("/scrape_url_to_markdown")
async def scrape_url_to_markdown(body: ScrapeRequest):
    try:
        result = await asyncio.wait_for(
            scrape_external_url_to_markdown(
                body.url, 
                body.force_rebuild, 
                body.include_metadata
            ),
            timeout=40
        )
        
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])

        return result

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Scraping timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))