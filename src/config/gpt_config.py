import os
from enum import Enum

class GPTMessageField(Enum):
    ROLE = "role"
    CONTENT = "content"

class GPTMessageRole(Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"

def get_gpt_config():
    gpt_model = os.getenv("OPENAI_GPT_MODEL", "gpt-4o-mini")
    gpt_api_key = os.getenv("OPENAI_API_KEY", "")
    if not gpt_model:
        raise ValueError("Could not identify what GPT to use.")
    if not gpt_api_key:
        raise ValueError("Could not get the GPT API key.")
    return {
        "model": gpt_model,
        "api_key": gpt_api_key,
    }
