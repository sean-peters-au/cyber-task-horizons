#!/usr/bin/env python3
"""
Simple proxy server that converts chat/completions requests to completions requests for davinci-002.

This allows inspect_ai to use davinci-002 by treating it as a "local" model.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import openai
import uvicorn

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment variables
load_dotenv()

app = FastAPI(title="Davinci-002 Proxy Server")

# OpenAI client for making real API calls
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not found in environment")
    sys.exit(1)

openai_client = openai.OpenAI(api_key=api_key)

def convert_chat_to_completion_prompt(messages):
    """Convert chat messages to a completion prompt."""
    prompt_parts = []
    
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role == "system":
            # System message becomes part of the prompt context
            prompt_parts.append(content)
        elif role == "user":
            # User message is the main prompt
            prompt_parts.append(content)
        elif role == "assistant":
            # Assistant message (for few-shot examples)
            prompt_parts.append(content)
    
    # Join with appropriate separators
    return "\n\n".join(prompt_parts)

def convert_completion_to_chat_response(completion_response):
    """Convert completion response to chat response format."""
    completion_text = completion_response.choices[0].text
    
    # Create chat-style response
    chat_response = {
        "id": f"chatcmpl-{completion_response.id}",
        "object": "chat.completion",
        "created": completion_response.created,
        "model": completion_response.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion_text
                },
                "finish_reason": completion_response.choices[0].finish_reason
            }
        ],
        "usage": completion_response.usage.model_dump() if completion_response.usage else None
    }
    
    return chat_response

@app.get("/v1/models")
async def list_models():
    """Return available models (required for validation)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "davinci-002",
                "object": "model",
                "created": 1649358449,
                "owned_by": "openai"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request_data: dict):
    """Convert chat/completions request to completions request."""
    try:
        # Extract parameters from chat request
        model = request_data.get("model", "davinci-002")
        messages = request_data.get("messages", [])
        max_tokens = request_data.get("max_tokens", 50)
        temperature = request_data.get("temperature", 0.0)
        stop = request_data.get("stop", [])
        
        # Convert chat messages to completion prompt
        prompt = convert_chat_to_completion_prompt(messages)
        
        print(f"🔄 Converting chat request for model: {model}")
        print(f"📝 Prompt: {prompt[:100]}...")
        print(f"🛑 Stop sequences received: {stop} (length: {len(stop) if isinstance(stop, list) else 'not a list'})")
        
        # Limit stop sequences to 4 (OpenAI's limit)
        if isinstance(stop, list) and len(stop) > 4:
            original_stop = stop.copy()
            stop = stop[:4]
            print(f"⚠️  Truncated stop sequences from {len(original_stop)} to {len(stop)}: {stop}")
        elif not stop:
            stop = ["\n", "\n\n"]
            print(f"📝 Using default stop sequences: {stop}")
        else:
            print(f"✅ Stop sequences OK: {stop}")
        
        # Make completion request to OpenAI
        completion_response = openai_client.completions.create(
            model="davinci-002",  # Always use davinci-002
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop
        )
        
        # Convert back to chat format
        chat_response = convert_completion_to_chat_response(completion_response)
        
        print(f"✅ Response: {completion_response.choices[0].text.strip()}")
        
        return JSONResponse(content=chat_response)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": "davinci-002"}

if __name__ == "__main__":
    print("🚀 Starting Davinci-002 Proxy Server on http://localhost:8001")
    print("🔧 This server converts chat/completions requests to completions requests")
    print("📡 Press Ctrl+C to stop")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )