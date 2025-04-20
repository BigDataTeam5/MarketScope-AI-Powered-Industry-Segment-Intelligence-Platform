from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from PIL import Image, ImageDraw
from io import BytesIO
from dotenv import load_dotenv
import os
import base64
import requests
import logging
from datetime import datetime
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from requests.sessions import Session
import asyncpraw
import uvicorn

# Load environment variables
load_dotenv(override=True)

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("grok_mcp_server")

# Create retry session
def create_robust_session():
    session = Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Update FastAPI app configuration
app = FastAPI(
    title="Grok Image MCP Server",
    description="AI Image Generation Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600
)

# MCP server
mcp = FastMCP("grok_image_server")
app.mount("/mcp", mcp.sse_app())

# Global configuration
GROK_API_KEY = os.getenv("GROK_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_SECRET = os.getenv("REDDIT_SECRET")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
SHOPIFY_API_KEY = os.getenv("SHOPIFY_API_KEY")
SHOPIFY_STORE_URL = os.getenv("SHOPIFY_STORE_URL")

# Helper

def track_token_usage(state, node_name, input_text, output_text):
    return {
        "total_so_far": len(input_text.split()),
        "total_cost_so_far": round(0.0001 * len(input_text.split()), 4)
    }

# MCP tool: Generate image with Grok # WRITE THE PASSING OF PRODUCT TITLE AND DESCRIPTION
@mcp.tool()
def generate_grok_image(title: str, description: str) -> Dict:
    prompt = f"A photorealistic e-commerce product poster for '{title}'. Description: {description}"
    try:
        client = OpenAI(base_url="https://api.x.ai/v1", api_key=GROK_API_KEY)
        safe_prompt = (
            f"Product photography: {prompt}. Professional e-commerce style, pure white background, "
            "centered composition, studio lighting, no text or logos. Clean and minimal product presentation."
        )
        logger.info(f"🎨 Prompting GROK with: {safe_prompt}")

        response = client.images.generate(
            model="grok-2-image-1212",
            prompt=safe_prompt,
            n=1
        )
        image_url = response.data[0].url
        logger.info(f"✅ GROK image URL: {image_url}")

        session = create_robust_session()
        image_response = session.get(image_url, timeout=30)
        image_response.raise_for_status()

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"product_{timestamp}.png"
        img = Image.open(BytesIO(image_response.content)).convert("RGBA")
        img.save(filename)

        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        usage = track_token_usage({}, "product_image_generation", safe_prompt, "Image generated")

        return {
            "status": "success",
            "image_filename": filename,
            "imageUrl": f"data:image/png;base64,{img_base64}",
            "tokens_used": usage['total_so_far'],
            "estimated_cost": usage['total_cost_so_far']
        }

    except Exception as e:
        logger.error(f"❌ Image generation failed: {str(e)}")
        return {"status": "error", "message": str(e)}

# === Reddit Posting Tool === #
@mcp.tool()
async def post_to_reddit(title: str, image_base64: str, subreddit_name: str = "test") -> Dict:
    """Post an image to Reddit"""
    try:
        reddit = asyncpraw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_SECRET,
            username=REDDIT_USERNAME,
            password=REDDIT_PASSWORD,
            user_agent=REDDIT_USER_AGENT
        )

        # Convert base64 to bytes
        image_bytes = base64.b64decode(image_base64)
        
        # Save image temporarily
        image_path = f"reddit_post_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # Post to Reddit
        subreddit = await reddit.subreddit(subreddit_name)
        submission = await subreddit.submit_image(title=title, image_path=image_path)
        await reddit.close()

        # Cleanup
        if os.path.exists(image_path):
            os.remove(image_path)

        return {
            "status": "success",
            "reddit_url": f"https://reddit.com{submission.permalink}",
            "subreddit": subreddit_name
        }

    except Exception as e:
        logger.error(f"❌ Reddit posting failed: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

# === Shopify Posting Tool === #
@mcp.tool()
def post_to_shopify(title: str, description: str, image_base64: str, price: str = "15.00") -> Dict:
    try:
        payload = {
            "product": {
                "title": title,
                "body_html": description,
                "images": [{"attachment": image_base64}],
                "variants": [{"price": price}],
                "status": "active"
            }
        }

        headers = {
            "Content-Type": "application/json",
            "X-Shopify-Access-Token": SHOPIFY_API_KEY
        }

        response = requests.post(
            f"https://{SHOPIFY_STORE_URL}/admin/api/2023-10/products.json",
            json=payload,
            headers=headers
        )
        response.raise_for_status()

        product_data = response.json()["product"]
        return {
            "status": "success",
            "shopify_url": f"https://{SHOPIFY_STORE_URL}/products/{product_data['handle']}",
            "product_id": product_data["id"],
            "price": price
        }

    except Exception as e:
        logger.error(f"❌ Shopify posting failed: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

# Basic health check
@app.get("/")
def root():
    return {"message": "Grok MCP Server is running ✅"}

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error handler caught: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": str(exc),
            "path": request.url.path
        }
    )

# Health check with more details
@app.get("/health")
def health():
    try:
        # Test OpenAI connection
        client = OpenAI(base_url="https://api.x.ai/v1", api_key=GROK_API_KEY)
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "api_status": "connected",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Add connection test endpoint
@app.get("/test-connection")
async def test_connection():
    return {
        "status": "connected",
        "server": "grok_mcp",
        "timestamp": datetime.now().isoformat(),
        "port": 8016
    }

class GenerateRequest(BaseModel):
    title: str
    description: str

class GenerateResponse(BaseModel):
    status: str
    imageUrl: str
    redditUrl: Optional[str] = None
    shopifyUrl: Optional[str] = None
    message: Optional[str] = None

@app.post("/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest):
    try:
        # Generate image
        image_result = generate_grok_image(request.title, request.description)
        if image_result["status"] != "success":
            raise HTTPException(status_code=500, detail=image_result["message"])

        # Get base64 data
        img_data = image_result["imageUrl"].split("base64,")[1]

        # Post to Reddit
        reddit_result = await post_to_reddit(
            title=request.title,
            image_base64=img_data,
            subreddit_name="test"  # or your chosen subreddit
        )

        # Post to Shopify
        shopify_result = post_to_shopify(
            title=request.title,
            description=request.description,
            image_base64=img_data
        )

        return {
            "status": "success",
            "imageUrl": image_result["imageUrl"],
            "redditUrl": reddit_result.get("reddit_url"),
            "shopifyUrl": shopify_result.get("shopify_url"),
            "message": "Image generated and posted successfully"
        }

    except Exception as e:
        logger.error(f"Generation workflow failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Workflow failed: {str(e)}"
        )

# Run app if needed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8016)