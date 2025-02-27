import os
import requests
from fastapi import FastAPI, Request, Response, HTTPException, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from urllib.parse import urljoin
import logging
from dotenv import load_dotenv
import uvicorn
import json

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set debug mode from environment variable
DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')

# Configure requests library logging when in debug mode
if DEBUG:
    # Enable HTTP request logging from the requests library
    requests_logger = logging.getLogger('urllib3')
    requests_logger.setLevel(logging.DEBUG)
    requests_logger.propagate = True

app = FastAPI(
    title="OpenAI API Proxy",
    description="A proxy server for OpenAI API that routes requests through a specified HTTP proxy",
    version="1.0.0"
)

# OpenAI API base URL
OPENAI_API_BASE_URL = os.getenv('OPENAI_API_BASE_URL', 'https://api.openai.com/')

# Proxy configuration
PROXY_CONFIG = None
http_proxy = os.getenv('HTTP_PROXY')
https_proxy = os.getenv('HTTPS_PROXY')

if http_proxy:
    PROXY_CONFIG = {
        "http": http_proxy,
        "https": https_proxy or http_proxy  # Use HTTP proxy for HTTPS if HTTPS proxy not specified
    }
    logger.info("Proxy configuration enabled")

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy(request: Request, path: str):
    """
    Proxy all requests to OpenAI API through the specified HTTP proxy.
    
    Args:
        request: The incoming FastAPI request
        path: The path part of the URL that will be appended to the base URL
        
    Returns:
        StreamingResponse: The proxied response from the OpenAI API
        
    Raises:
        HTTPException: If there's an error during the proxying process
    """
    # Construct target URL
    target_url = urljoin(OPENAI_API_BASE_URL, path)
    logger.info(f"Proxying request to: {target_url}")
    
    # Get request method
    method = request.method
    
    # Get headers from the original request
    headers = dict(request.headers)
    if "host" in headers:
        del headers["host"]
    
    # Get query parameters
    params = dict(request.query_params)
    
    # Log detailed request information in debug mode
    if DEBUG:
        debug_info = {
            "method": method,
            "target_url": target_url,
            "headers": {k: v for k, v in headers.items() if k.lower() not in ('authorization')},  # Don't log auth tokens
            "params": params
        }
        logger.debug(f"Request details: {json.dumps(debug_info, indent=2)}")
    
    try:
        # Common request kwargs
        request_kwargs = {
            "headers": headers,
            "params": params,
            "stream": True
        }
        
        # Add proxy configuration if enabled
        if PROXY_CONFIG:
            request_kwargs["proxies"] = PROXY_CONFIG
        
        # Handle different request methods and content types
        if method == 'GET':
            response = requests.get(target_url, **request_kwargs)
        else:  # POST, PUT, DELETE, etc.
            content_type = headers.get('Content-Type', '')
            
            if 'multipart/form-data' in content_type:
                # For multipart form data, we need to handle it differently
                form_data = await request.form()
                files = {}
                data = {}
                
                # Process form data
                for key, value in form_data.items():
                    if isinstance(value, UploadFile):
                        # Handle file uploads
                        file_content = await value.read()
                        files[key] = (value.filename, file_content, value.content_type)
                    else:
                        # Handle regular form fields
                        data[key] = value
                
                request_kwargs.update({
                    "data": data,
                    "files": files
                })
                
                if DEBUG:
                    logger.debug(f"Form data: {data}")
                    logger.debug(f"Files: {[f for f in files.keys()]}")
                
                response = requests.request(method, target_url, **request_kwargs)
            else:
                # Handle JSON or other content types
                body = await request.body()
                request_kwargs["data"] = body
                
                if DEBUG and body:
                    try:
                        # Try to parse and log the body if it's JSON
                        if 'application/json' in content_type:
                            body_json = json.loads(body)
                            # Redact sensitive fields if present
                            if 'messages' in body_json:
                                logger.debug(f"Request body contains {len(body_json['messages'])} messages")
                            else:
                                logger.debug(f"Request body: {json.dumps(body_json, indent=2)}")
                        else:
                            logger.debug(f"Request body length: {len(body)}")
                    except:
                        logger.debug(f"Request body length: {len(body)}")
                
                response = requests.request(method, target_url, **request_kwargs)
        
        # Log response details in debug mode
        if DEBUG:
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
        
        # Create a streaming response with the same status code and headers
        def generate_stream():
            for chunk in response.iter_content(chunk_size=1024):
                if DEBUG:
                    logger.debug(f"Streaming chunk: {len(chunk)} bytes")
                yield chunk
        
        # Create response headers
        response_headers = {}
        for key, value in response.headers.items():
            if key.lower() not in ('content-encoding', 'transfer-encoding', 'content-length'):
                response_headers[key] = value
        
        return StreamingResponse(
            generate_stream(),
            status_code=response.status_code,
            headers=response_headers,
            media_type=response.headers.get('Content-Type')
        )
        
    except Exception as e:
        logger.error(f"Error proxying request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    host = os.getenv('HOST', '0.0.0.0')
    
    logger.info(f"Starting OpenAI API Proxy on {host}:{port}")
    logger.info(f"Debug mode: {DEBUG}")
    logger.info(f"Proxy enabled: {PROXY_CONFIG is not None}")
    
    uvicorn.run("main:app", host=host, port=port, reload=DEBUG)
