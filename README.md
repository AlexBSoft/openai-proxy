# OpenAI API Proxy

A lightweight reverse proxy built with FastAPI that forwards requests to the OpenAI API, optionally through an HTTP proxy.

## Features

- Proxies all requests to the OpenAI API (configurable via environment variable)
- Preserves all original request headers, methods, and body content
- Supports streaming responses (SSE) for chat completions
- Optionally routes traffic through an HTTP proxy if configured
- Built with FastAPI for high performance
- Minimal configuration required

## Setup

1. Configure environment variables:
   Copy the `.env.example` file to `.env` and modify as needed:
   ```
   cp .env.example .env
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python main.py
   ```

4. The proxy server will start on port 8080 by default (configurable via PORT environment variable)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| OPENAI_API_BASE_URL | Base URL for the OpenAI API | https://api.openai.com/ |
| HTTP_PROXY | HTTP proxy URL (optional) | None |
| PORT | Port to run the server on | 8080 |
| DEBUG | Enable debug logging | False |

## Usage

Send your OpenAI API requests to this proxy instead of directly to OpenAI.

Example:
```bash
# Instead of
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello!"}]}'

# Use
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Docker

You can also run the proxy using Docker:

```bash
# Build the Docker image
docker build -t openai-proxy .

# Run the container
docker run -p 8080:8080 --env-file .env openai-proxy
```

## Notes

- If HTTP_PROXY is set in the environment, all requests to the OpenAI API will be routed through this proxy
- The proxy preserves all headers and request parameters
- Streaming responses (like those from chat completions) are properly handled 