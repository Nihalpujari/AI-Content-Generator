from __future__ import annotations
import os
import io
from typing import Any, Dict, Tuple


class ImageGenerator:
    """
    Free image generator using Hugging Face InferenceClient.
    """

    def __init__(self):
        self.api_key = (os.getenv("HUGGINGFACE_API_KEY") or "").strip()
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY missing in .env")
        self.model = "black-forest-labs/FLUX.1-schnell"

    def generate(self, *, prompt: str, size: str = "1024x1024") -> Tuple[bytes, Dict[str, Any]]:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=self.api_key)
        image = client.text_to_image(prompt, model=self.model)

        # Convert PIL image to bytes
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        meta: Dict[str, Any] = {
            "model": self.model,
            "size": size,
            "encoding": "bytes",
        }

        return img_bytes, meta