# -*- coding: utf-8 -*-
"""
GeminiProvider - Google Gemini API implementation.

This provider wraps the google-genai SDK to provide image-based content generation.
When using a custom base URL (e.g., corporate proxy), it falls back to OpenAI-compatible format.
"""
import asyncio
import base64
import json
import os
from pathlib import Path
from typing import List, Type, TypeVar, cast

from google import genai
from google.genai import types
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.models import THINKING_LEVEL_MODELS

ResponseT = TypeVar("ResponseT", bound=BaseModel)


def extract_first_json_block(text: str) -> dict | None:
    """Extract the first JSON code block from text."""
    import re

    pattern = r"```json\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)
    if matches:
        return json.loads(matches[0])
    return None


class GeminiProvider:
    """
    Gemini-based chat provider implementation.

    This class encapsulates all Gemini-specific logic, making it easy to
    swap out for other providers in the future.
    """

    def __init__(self, api_key: str, model: str):
        """
        Initialize the Gemini provider.

        Args:
            api_key: Gemini API key.
            model: Model name to use (e.g., "gemini-2.5-pro").
        """
        self._api_key = api_key
        self._model = model
        self._client: genai.Client | None = None
        self._response: types.GenerateContentResponse | None = None
        self._use_openai_format = self._should_use_openai_format()

    def _should_use_openai_format(self) -> bool:
        """Check if we should use OpenAI-compatible format instead of native Gemini format."""
        # Use OpenAI format when custom base URL is set (e.g., corporate proxies)
        return bool(os.getenv("GOOGLE_GEMINI_BASE_URL"))

    @property
    def client(self) -> genai.Client:
        """Lazy-initialize the Gemini client."""
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    @property
    def last_response(self) -> types.GenerateContentResponse | None:
        """Get the last response for debugging/caching purposes."""
        return self._response

    async def _upload_files(self, files: List[Path]) -> list[types.File]:
        """Upload multiple files concurrently."""
        valid_files = [f for f in files if f and Path(f).exists()]
        if not valid_files:
            return []
        upload_tasks = [self.client.aio.files.upload(file=f) for f in valid_files]
        return list(await asyncio.gather(*upload_tasks))

    @staticmethod
    def _files_to_parts(files: List[types.File]) -> List[types.Part]:
        """Convert uploaded files to parts."""
        return [types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type) for f in files]

    def _set_thinking_config(self, config: types.GenerateContentConfig) -> None:
        """Configure thinking settings based on model capabilities."""
        config.thinking_config = types.ThinkingConfig(include_thoughts=True)

        if self._model in THINKING_LEVEL_MODELS:
            thinking_level = types.ThinkingLevel.HIGH

            config.thinking_config = types.ThinkingConfig(
                include_thoughts=False, thinking_level=thinking_level
            )

    async def _generate_openai_format(
        self,
        *,
        images: List[Path],
        response_schema: Type[ResponseT],
        user_prompt: str | None = None,
        description: str | None = None,
        **kwargs,
    ) -> ResponseT:
        """Generate content using OpenAI-compatible format (for proxies)."""
        import httpx

        base_url = os.getenv("GOOGLE_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
        endpoint = f"{base_url}/v1/chat/completions"

        # Build message content with images
        content = []

        # Add images as base64 data URLs
        image_sizes = []
        for image_path in images:
            if not image_path.exists():
                continue

            # Detect MIME type
            import mimetypes
            mime_type, _ = mimetypes.guess_type(str(image_path))
            if not mime_type or not mime_type.startswith('image/'):
                mime_type = 'image/png'

            # Read and encode
            with open(image_path, 'rb') as f:
                image_data = f.read()
                image_sizes.append(len(image_data))
                image_b64 = base64.b64encode(image_data).decode('utf-8')

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_b64}"
                }
            })

        # Add text prompt
        prompt_preview = user_prompt[:100] + "..." if user_prompt and len(user_prompt) > 100 else user_prompt
        if user_prompt:
            content.append({
                "type": "text",
                "text": user_prompt
            })

        # Build request payload
        payload = {
            "model": self._model,
            "messages": []
        }

        # Add system message if description provided
        if description:
            payload["messages"].append({
                "role": "system",
                "content": description
            })

        # Add user message with images and text
        payload["messages"].append({
            "role": "user",
            "content": content
        })

        # Add response format for structured output
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": response_schema.model_json_schema()
            }
        }

        # Set temperature and max_tokens
        payload["temperature"] = 0.1
        payload["max_tokens"] = 8192

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }

        # Log request details
        total_image_size = sum(image_sizes)
        logger.info(f"🚀 OpenAI API Request: model={self._model}, images={len(image_sizes)}, "
                   f"total_size={total_image_size/1024:.1f}KB, endpoint={endpoint}")
        if prompt_preview:
            logger.debug(f"   Prompt: {prompt_preview}")

        # Make request
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(endpoint, json=payload, headers=headers)

                # Log HTTP status
                logger.info(f"✅ HTTP {response.status_code} - Response received from {endpoint}")

                response.raise_for_status()
                data = response.json()

                # Log usage statistics if available
                if "usage" in data:
                    usage = data["usage"]
                    logger.info(f"📊 Tokens: prompt={usage.get('prompt_tokens', 0)}, "
                               f"completion={usage.get('completion_tokens', 0)}, "
                               f"total={usage.get('total_tokens', 0)}")

        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP {e.response.status_code} Error from {endpoint}")
            logger.error(f"   Response: {e.response.text[:500]}")
            raise
        except Exception as e:
            logger.error(f"❌ Request failed: {type(e).__name__}: {e}")
            raise

        # Parse OpenAI response
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                content_text = choice["message"]["content"]

                # Try to parse as JSON
                try:
                    if content_text:
                        json_data = json.loads(content_text)
                        logger.info(f"✅ Successfully parsed structured response")
                        return response_schema(**json_data)
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️  JSON parse failed, trying to extract JSON block: {e}")
                    # Try to extract JSON block
                    json_data = extract_first_json_block(content_text)
                    if json_data:
                        logger.info(f"✅ Extracted JSON from code block")
                        return response_schema(**json_data)

        logger.error(f"❌ Failed to parse response structure: {str(data)[:500]}")
        raise ValueError(f"Failed to parse OpenAI response: {data}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry request ({retry_state.attempt_number}/3) - "
            f"Wait 3 seconds - Exception: {retry_state.outcome.exception()}"
        ),
    )
    async def generate_with_images(
        self,
        *,
        images: List[Path],
        response_schema: Type[ResponseT],
        user_prompt: str | None = None,
        description: str | None = None,
        **kwargs,
    ) -> ResponseT:
        """
        Generate content with image inputs.

        Args:
            images: List of image file paths to include in the request.
            user_prompt: User-provided prompt/instructions.
            description: System instruction/description for the model.
            response_schema: Pydantic model class for structured output.
            **kwargs: Additional options passed to the API.

        Returns:
            Parsed response matching the response_schema type.
        """
        # Route to OpenAI format if using custom base URL (e.g., Huya proxy)
        if self._use_openai_format:
            logger.info("Using OpenAI-compatible format for custom base URL")
            return await self._generate_openai_format(
                images=images,
                response_schema=response_schema,
                user_prompt=user_prompt,
                description=description,
                **kwargs
            )

        # Otherwise use native Gemini SDK format
        uploaded_files = await self._upload_files(images)
        parts = self._files_to_parts(uploaded_files)

        logger.info(f"🚀 Native Gemini API Request: model={self._model}, files={len(uploaded_files)}")

        # Add user prompt if provided
        if user_prompt and isinstance(user_prompt, str):
            parts.append(types.Part.from_text(text=user_prompt))

        contents = [types.Content(role="user", parts=parts)]

        # Build config
        config = types.GenerateContentConfig(
            system_instruction=description,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        # Set thinking config if applicable
        self._set_thinking_config(config=config)

        # Generate response
        try:
            self._response: types.GenerateContentResponse = (
                await self.client.aio.models.generate_content(
                    model=self._model, contents=contents, config=config
                )
            )
            logger.info(f"✅ Response received from native Gemini API")
        except Exception as e:
            logger.error(f"❌ Native Gemini API request failed: {type(e).__name__}: {e}")
            raise

        # Parse response
        if self._response.parsed:
            parsed = self._response.parsed
            if isinstance(parsed, BaseModel):
                logger.info(f"✅ Successfully parsed structured response")
                return response_schema(**parsed.model_dump())
            if isinstance(parsed, dict):
                logger.info(f"✅ Successfully parsed structured response")
                return response_schema(**cast(dict[str, object], parsed))

        # Fallback to JSON extraction
        if response_text := self._response.text:
            json_data = extract_first_json_block(response_text)
            if json_data:
                logger.info(f"✅ Extracted JSON from code block")
                return response_schema(**json_data)

        logger.error(f"❌ Failed to parse response: {self._response.text[:500] if self._response.text else 'empty'}")
        raise ValueError(f"Failed to parse response: {self._response.text}")

    def cache_response(self, path: Path) -> None:
        """Cache the last response to a file."""
        if not self._response:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._response.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
