"""LLM initialization module using LangChain built-in chat models.

This module strictly leverages:
- `langchain_core.language_models.chat_models.BaseChatModel`: Base chat model abstraction.
- `langchain_google_genai.ChatGoogleGenerativeAI`: Official Google Gemini chat model.
- `langchain_openai.ChatOpenAI`: Official OpenAI chat model.
- `langchain_core.language_models.fake_chat_models.FakeListChatModel`: Mock chat model for unit tests.
"""

import os
from typing import Any, Optional
from dotenv import load_dotenv

from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()


def get_chat_model(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Initialize a LangChain Chat Model based on environment and provider.

    Args:
        model: Model name/identifier (e.g. 'gemini-3.5-flash-lite' or 'gpt-4o-mini').
        provider: 'google', 'openai', or 'fake'. Auto-detected if None.
        temperature: Sampling temperature (optional).
        **kwargs: Additional parameters passed to the chat model.

    Returns:
        BaseChatModel: An instance of LangChain BaseChatModel.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    resolved_provider = (
        provider
        or os.getenv("LLM_PROVIDER")
        or ("google" if gemini_key else "openai" if openai_key else "fake")
    ).lower()

    if resolved_provider in ("google", "gemini", "google_genai"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = (
            model
            or os.getenv("GEMINI_DEFAULT_MODEL")
            or "gemini-3.5-flash-lite"
        )
        chat_kwargs = {"model": model_name, "google_api_key": gemini_key, **kwargs}
        if temperature is not None:
            chat_kwargs["temperature"] = temperature
        return ChatGoogleGenerativeAI(**chat_kwargs)

    elif resolved_provider == "openai":
        from langchain_openai import ChatOpenAI

        model_name = model or os.getenv("OPENAI_DEFAULT_MODEL") or "gpt-4o-mini"
        chat_kwargs = {"model": model_name, "api_key": openai_key, **kwargs}
        if temperature is not None:
            chat_kwargs["temperature"] = temperature
        return ChatOpenAI(**chat_kwargs)

    elif resolved_provider == "fake":
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        responses = kwargs.get(
            "responses",
            [
                "Docker is a containerization platform that isolates applications.",
                "FastAPI is a modern, fast web framework for building APIs in Python.",
                "This is a general query response from the AI assistant.",
            ],
        )
        return FakeListChatModel(responses=responses)

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{resolved_provider}'. "
            "Supported providers: 'google', 'openai', 'fake'."
        )
