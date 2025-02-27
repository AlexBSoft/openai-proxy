# OpenAI API Proxy

A modern web application built with FastAPI that proxies requests to the OpenAI API through a specified HTTP proxy.

## Features

- Proxies all requests to `https://api.openai.com/`
- Preserves headers, request methods (GET, POST, etc.)
- Supports JSON body and multipart form data with binary files
- Uses a specified HTTP proxy
- Built with FastAPI for high performance
- Automatic API documentation at `/docs` and `/redoc`

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python main.py
   ```

3. The proxy server will start on `http://localhost:5000` by default
4. Access the API documentation at `http://localhost:5000/docs`

## Usage

Send your OpenAI API requests to this proxy instead of directly to OpenAI.

Example:
```
# Instead of
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello!"}]}'

# Use
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello!"}]}'
```

All requests will be proxied through the configured HTTP proxy. 