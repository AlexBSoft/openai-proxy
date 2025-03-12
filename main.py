import os
import logging
from typing import Any, Dict, Optional
from fastapi import FastAPI, Form, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import requests
from dotenv import load_dotenv
import io
from requests_toolbelt.multipart.encoder import MultipartEncoder

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=(
        logging.INFO
        if os.getenv("DEBUG", "False").lower() == "true"
        else logging.WARNING
    ),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Get environment variables
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/")
HTTP_PROXY = os.getenv("HTTP_PROXY")
PORT = int(os.getenv("PORT", 8080))

# Ensure OPENAI_API_BASE_URL ends with a slash
if not OPENAI_API_BASE_URL.endswith("/"):
    OPENAI_API_BASE_URL += "/"

# Configure proxy settings if HTTP_PROXY is set
proxies = {}
if HTTP_PROXY:
    proxies = {"http": HTTP_PROXY, "https": HTTP_PROXY}
    logger.info(f"Using HTTP proxy: {HTTP_PROXY}")

app = FastAPI(title="OpenAI API Proxy")


@app.get("/")
async def root():
    return {
        "message": "OpenAI API Proxy is running. Send your OpenAI API requests to this server."
    }


async def handle_audio_transcription(request: Request):
    """
    Handle audio transcription requests to the OpenAI API.
    """
    # Construct the target URL
    target_url = f"{OPENAI_API_BASE_URL}v1/audio/transcriptions"

    # Get request headers
    headers = dict(request.headers)
    # Remove host header as it will be set by the requests library
    headers.pop("host", None)

    logger.info("Handling audio transcription request")

    # Parse the multipart form data
    form = await request.form()

    # Extract the file and model parameter
    audio_file = None
    model = "whisper-1"  # Default model

    for field_name, field_value in form.items():
        if field_name == "file" and hasattr(field_value, "filename"):
            # Get the audio file
            audio_file = field_value
            logger.info(f"Found audio file: {field_value.filename}")
        elif field_name == "model":
            # Get the model parameter
            model = str(field_value)
            logger.info(f"Using model: {model}")

    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided")

    # Read the file content
    file_content = await audio_file.read()

    # Create a multipart encoder
    mp_encoder = MultipartEncoder(
        fields={
            "file": (
                audio_file.filename,
                io.BytesIO(file_content),
                audio_file.content_type or "audio/mpeg",
            ),
            "model": model,
        }
    )

    # Update the content-type header
    headers["Content-Type"] = mp_encoder.content_type

    # Make the request to OpenAI
    logger.info(f"Sending request to {target_url}")

    try:
        response = requests.post(
            url=target_url,
            headers=headers,
            data=mp_encoder,
            proxies=proxies if HTTP_PROXY else None,
            timeout=60,
        )

        # Log the response for debugging
        logger.info(f"Response status: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Error response: {response.text}")

        # Return the response
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )
    except requests.RequestException as e:
        logger.error(f"Error proxying transcription request: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error proxying transcription request: {str(e)}"
        )


# Register the dedicated endpoint for audio transcriptions
@app.post("/v1/audio/transcriptions")
async def audio_transcriptions_endpoint(request: Request):
    return await handle_audio_transcription(request)


@app.post("/v1/evelabs/tts")
async def elevenlabs_tts_proxy(
    text: str = Form(...),
    voice_id: str = Form(...),
    ELEVENLABS_API_KEY: str = Form(...),
    model_id: str = Form("eleven_multilingual_v2"),
    output_format: str = Form("mp3_44100_128"),
):
    """
    Прокси для ElevenLabs Text-to-Speech API.
    """

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format={output_format}"

    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}

    payload = {
        "text": text,
        "model_id": model_id,
        # "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
        # "output_format": output_format,
    }

    try:
        response = requests.post(
            url=url,
            json=payload,
            headers=headers,
            proxies=proxies,
            stream=False,
            timeout=60,
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"ElevenLabs error: {response.text}",
            )

        # Возвращаем байты ауидо файла
        return Response(content=response.content)

    except requests.RequestException as e:
        raise HTTPException(
            status_code=500, detail=f"Error proxying to ElevenLabs: {str(e)}"
        )


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    include_in_schema=False,
)
async def proxy(request: Request, path: str):
    # Construct the target URL
    target_url = f"{OPENAI_API_BASE_URL}{path}"

    # Get request headers
    headers = dict(request.headers)
    # Remove host header as it will be set by the requests library
    headers.pop("host", None)

    # Get query parameters
    params = dict(request.query_params)

    try:
        # Handle different content types appropriately
        content_type = request.headers.get("content-type", "")

        if "multipart/form-data" in content_type:
            # For multipart/form-data, we need to handle file uploads
            form_data = await request.form()
            files = {}
            data = {}

            for field_name, field_value in form_data.items():
                if hasattr(field_value, "filename") and field_value.filename:
                    # This is a file
                    # Instead of reading the file content, pass the file object directly
                    # This preserves the file's content type and other metadata
                    file_content = await field_value.read()
                    content_type = (
                        field_value.content_type or "application/octet-stream"
                    )
                    files[field_name] = (
                        field_value.filename,
                        file_content,
                        content_type,
                    )
                    logger.info(
                        f"Processing file upload: {field_name}={field_value.filename} ({content_type})"
                    )
                else:
                    # This is a regular form field
                    data[field_name] = str(field_value)
                    logger.info(f"Processing form field: {field_name}={field_value}")

            # Log the request details for debugging
            logger.info(f"Sending multipart request to {target_url}")
            logger.info(f"Files: {[f'{k}={v[0]}' for k, v in files.items()]}")
            logger.info(f"Data: {data}")

            # Make the request with files and form data
            response = requests.request(
                method=request.method,
                url=target_url,
                headers=headers,
                params=params,
                data=data,
                files=files,
                proxies=proxies if HTTP_PROXY else None,
                stream=True,
                verify=True,
                allow_redirects=True,
                timeout=60,
            )
        else:
            # For other content types, use the original approach
            body = await request.body()
            response = requests.request(
                method=request.method,
                url=target_url,
                headers=headers,
                params=params,
                data=body,
                proxies=proxies if HTTP_PROXY else None,
                stream=True,
                verify=True,
                allow_redirects=True,
                timeout=60,
            )

        # If the response is streaming, return a streaming response
        if (
            "content-encoding" in response.headers
            or "transfer-encoding" in response.headers
        ):
            return StreamingResponse(
                response.iter_content(chunk_size=8192),
                status_code=response.status_code,
                headers=dict(response.headers),
            )

        # Otherwise, return a regular response
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )
    except requests.RequestException as e:
        logger.error(f"Error proxying request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error proxying request: {str(e)}")


if __name__ == "__main__":
    logger.info(f"Starting OpenAI API Proxy on port {PORT}")
    logger.info(f"Proxying requests to {OPENAI_API_BASE_URL}")
    if HTTP_PROXY:
        logger.info(f"Using HTTP proxy: {HTTP_PROXY}")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
