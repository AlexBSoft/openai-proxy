import os
import logging
from typing import Any, Dict, Optional
from fastapi import FastAPI, Form, Request, Response, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
import uvicorn
import requests
from dotenv import load_dotenv
import io
import httpx
from io import BytesIO
from requests_toolbelt.multipart.encoder import MultipartEncoder
import asyncio
import tempfile

# import subprocess
# import ffmpeg
# from urllib.parse import unquote


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
API_BASE_ZAPCAP = os.getenv("API_BASE_ZAPCAP")

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
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
        "output_format": output_format,
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
        # return Response(content=response.content)
        return Response(
            content=response.content,
            media_type="audio/mpeg",
            headers={"Content-Disposition": 'attachment; filename="output.mp3"'},
        )

    except requests.RequestException as e:
        raise HTTPException(
            status_code=500, detail=f"Error proxying to ElevenLabs: {str(e)}"
        )


async def add_captions(video_path: str, api_key: str, template_id: str) -> BytesIO:

    async with httpx.AsyncClient() as client:

        try:
            # 1. Upload video
            print("Uploading video...")
            with open(video_path, "rb") as f:
                upload_response = await client.post(
                    f"{API_BASE_ZAPCAP}/videos",
                    headers={"x-api-key": api_key},
                    files={"file": f},
                )
            upload_response.raise_for_status()
            video_id = upload_response.json()["id"]
            print("Video uploaded, ID:", video_id)

            # 2. Create task
            print("Creating captioning task...")
            task_response = await client.post(
                f"{API_BASE_ZAPCAP}/videos/{video_id}/task",
                headers={"x-api-key": api_key, "Content-Type": "application/json"},
                json={
                    "templateId": template_id,
                    "autoApprove": True,
                    "language": "ru",
                    "transcribeSettings": {"broll": {"brollPercent": 50}},
                },
            )
            task_response.raise_for_status()
            task_id = task_response.json()["taskId"]
            print("Task created, ID:", task_id)

            # 3. Poll for completion
            print("Processing video...")
            attempts = 0
            while True:
                status_response = await client.get(
                    f"{API_BASE_ZAPCAP}/videos/{video_id}/task/{task_id}",
                    headers={"x-api-key": api_key},
                )
                status_response.raise_for_status()
                data = status_response.json()
                status = data["status"]
                print("Status:", status)

                if status == "completed":
                    # Download the video
                    print("Downloading captioned video...")
                    download_response = await client.get(data["downloadUrl"])
                    download_response.raise_for_status()

                    # Return the captioned video as a BytesIO object
                    return BytesIO(download_response.content)
                elif status == "failed":
                    raise Exception(f"Task failed: {data.get('error')}")

                await asyncio.sleep(2)
                attempts += 1

        except Exception as e:
            print("Error:", str(e))
            raise e


@app.post("/zapcut/upload-video/")
async def upload_video(
    template_id: str,
    api_key: str,
    video_url: str,
):
    try:

        logging.info(f"[upload-video]:{video_url}")
        # Скачиваем видео по ссылке
        async with httpx.AsyncClient() as client:
            response = await client.get(video_url)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=404, detail="Video not found at the provided URL."
                )
            video_content = response.content

        # Сохраняем видео временно в файл
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as temp_file:
            temp_file.write(video_content)

        # Обрабатываем видео, добавляя субтитры
        try:
            processed_video = await add_captions(temp_video_path, api_key, template_id)
        except Exception as e:
            # Логируем ошибку и отправляем сообщение клиенту
            logging.ERROR(
                f"Error processing the video: {str(e),e.args,e.__traceback__}"
            )
            raise HTTPException(
                status_code=500,
                detail="Error processing the video. Please try again later.",
            )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(processed_video.getvalue())
            temp_file_path = temp_file.name  # Сохраняем путь к файлу

        # Удаляем временный файл после отправки
        os.remove(temp_file_path)
        # Удаляем временный файл
        os.remove(temp_video_path)

        # Возвращаем обработанное видео в виде потока
        return StreamingResponse(
            processed_video,  # Используем сам BytesIO объект
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename=captioned_video.mp4"
            },
        )

    except Exception as e:
        # Ловим все исключения при загрузке видео
        print(f"Error with video processing: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error downloading or processing the video. Please try again later.",
        )


# async def run_ffmpeg(*args):
#     """Запускает FFmpeg асинхронно"""
#     process = await asyncio.create_subprocess_exec(
#         "ffmpeg", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
#     )
#     await process.communicate()  # Дождаться завершения процесса


# @app.post("/ffmpeg/process-video/")
# async def process_video(
#     video: UploadFile = File(...),
#     audio: UploadFile = File(...),
# ):
#     try:
#         # Сохраняем видео и аудио во временные файлы
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_file:
#             video_file.write(await video.read())
#             video_path = video_file.name

#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
#             audio_file.write(await audio.read())
#             audio_path = audio_file.name

#         no_audio_video_path = os.path.join(tempfile.gettempdir(), "no_audio_video.mp4")
#         final_video_path = os.path.join(tempfile.gettempdir(), "final_video.mp4")

#         try:
#             # Удаляем аудио из видео (асинхронно)
#             await run_ffmpeg("-i", video_path, "-an", no_audio_video_path, "-y")

#             # Добавляем новую аудиодорожку (асинхронно)
#             await run_ffmpeg(
#                 "-i",
#                 no_audio_video_path,
#                 "-i",
#                 audio_path,
#                 "-c:v",
#                 "copy",
#                 "-c:a",
#                 "aac",
#                 final_video_path,
#                 "-y",
#             )

#             video_file = FSInputFile(final_video_path)
#             async with Bot(
#                 token="8116357102:AAGW7w6gvZFajhqFsH-aSJFBbWWKdKWQWIc"
#             ) as bot:
#                 await bot.send_document(
#                     chat_id=int(-1002255836822),
#                     caption="Готовое видео 🎥",
#                     document=video_file,
#                 )
#             # Открываем финальное видео и возвращаем его как поток
#             with open(final_video_path, "rb") as final_video_file:
#                 final_video = BytesIO(final_video_file.read())
#         except Exception as e:
#             print(f"Error during video processing: {str(e)}")
#             return {"error": f"Error processing video: {str(e)}"}

#         finally:
#             # Удаляем временные файлы
#             for file in [video_path, audio_path, no_audio_video_path, final_video_path]:
#                 if os.path.exists(file):
#                     os.remove(file)

#         return StreamingResponse(
#             final_video,
#             media_type="video/mp4",
#             headers={"Content-Disposition": "attachment; filename=processed_video.mp4"},
#         )
#     except Exception as e:
#         print(f"Error processing video: {str(e)}")
#         return {"error": f"Error processing video: {str(e)}"}


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
