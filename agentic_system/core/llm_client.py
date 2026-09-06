"""
llm_client.py
=============
LLM abstraction layer. Provides a single interface for all four supported
providers: OpenAI, Anthropic, HuggingFace, and Google.

All agents call LLMClient.complete(system_prompt, user_prompt) → str.
The underlying provider is determined entirely by config.yaml — no agent
code changes are needed when switching models.

Supported providers:
    openai       → GPT-4o, GPT-3.5-turbo, etc.
    anthropic    → claude-sonnet-4-6, claude-opus-4-6, etc.
    huggingface  → Mistral-7B, LLaMA-3-8B (via HF Inference API)
    google       → Gemini 1.5 Flash, Gemini 1.5 Pro

Author: Mohammed Mudassirullah Sheriff
Project: Intelligent Optimization of Data Pipelines using RAG-Based
         Knowledge Retrieval and Agentic Decision Systems
Institution: BITS Pilani WILP — M.Tech Data Science & Engineering
"""

import os
import time
import logging
from abc import ABC, abstractmethod

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = None) -> dict:
    """Load config.yaml and return as a dict."""
    if config_path is None:
        # Default: look for config.yaml relative to this file
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "config.yaml"
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM provider clients.
    Every provider must implement the complete() method.
    """

    def __init__(self, model: str, temperature: float, max_tokens: int):
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a prompt to the LLM and return the text response.

        Args:
            system_prompt : Role and instruction context for the model
            user_prompt   : The actual task or query

        Returns:
            str — raw text response from the model
        """
        pass

    def complete_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        """
        Wrapper around complete() with retry logic for transient errors.
        Used by all agents for robustness.
        """
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.complete(system_prompt, user_prompt)
                return response
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM call failed (attempt {attempt}/{max_retries}): {e}"
                )
                if attempt < max_retries:
                    time.sleep(retry_delay)

        raise RuntimeError(
            f"LLM call failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        )


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIClient(BaseLLMClient):
    """
    OpenAI provider client.
    Supports GPT-4o, GPT-4-turbo, GPT-3.5-turbo, and any OpenAI chat model.

    Install: pip install openai
    """

    def __init__(self, model: str, temperature: float, max_tokens: int, api_key: str):
        super().__init__(model, temperature, max_tokens)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class AnthropicClient(BaseLLMClient):
    """
    Anthropic provider client.
    Supports claude-sonnet-4-6, claude-opus-4-6, claude-haiku-4-5, etc.

    Install: pip install anthropic
    """

    def __init__(self, model: str, temperature: float, max_tokens: int, api_key: str):
        super().__init__(model, temperature, max_tokens)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# HuggingFace provider (Inference API)
# ---------------------------------------------------------------------------

class HuggingFaceClient(BaseLLMClient):
    """
    HuggingFace Inference API client.
    Supports Mistral-7B-Instruct, LLaMA-3-8B-Instruct, and other HF models.

    Note: Larger models may be slow on free-tier HF Inference API.
          Consider using a local Ollama server for faster local inference.

    Install: pip install huggingface_hub
    """

    def __init__(self, model: str, temperature: float, max_tokens: int, api_key: str):
        super().__init__(model, temperature, max_tokens)
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(model=model, token=api_key)
        except ImportError:
            raise ImportError(
                "huggingface_hub not installed. Run: pip install huggingface_hub"
            )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        # HF chat_completion follows the OpenAI message format
        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        choice = response.choices[0]
        logger.info("Finish reason : %s", getattr(choice, "finish_reason", None))

        if hasattr(response, "usage"):
            logger.info(
                "Prompt=%s Completion=%s Total=%s",
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens,
            )

        logger.info(
            "Output chars=%d",
            len(choice.message.content)
        )
        return choice.message.content.strip()


# ---------------------------------------------------------------------------
# Google Gemini provider
# ---------------------------------------------------------------------------

class GoogleClient(BaseLLMClient):
    """
    Google Gemini provider client.
    Supports gemini-1.5-flash, gemini-1.5-pro.

    Install: pip install google-generativeai
    """

    def __init__(self, model: str, temperature: float, max_tokens: int, api_key: str):
        super().__init__(model, temperature, max_tokens)
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.genai  = genai
            self.model_name = model
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Run: pip install google-generativeai"
            )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        model = self.genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
            generation_config=self.genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        response = model.generate_content(user_prompt)
        return response.text.strip()


# ---------------------------------------------------------------------------
# Factory — builds the correct client from config
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Factory class. Instantiate this in agent code — never instantiate
    provider clients directly.

    Usage:
        client = LLMClient()
        response = client.complete(system_prompt, user_prompt)

    Or with retry:
        response = client.complete_with_retry(system_prompt, user_prompt)
    """

    PROVIDER_MAP = {
        "openai":       OpenAIClient,
        "anthropic":    AnthropicClient,
        "huggingface":  HuggingFaceClient,
        "google":       GoogleClient,
    }

    def __init__(self, config_path: str = None):
        config      = load_config(config_path)
        llm_config  = config["llm"]

        provider    = llm_config["provider"].lower()
        model       = llm_config["model"]
        temperature = float(llm_config.get("temperature", 0.2))
        max_tokens  = int(llm_config.get("max_tokens", 1500))
        api_key_env = llm_config.get("api_key_env", "")
        api_key     = os.environ.get(api_key_env, "")

        if not api_key and provider != "huggingface":
            logger.warning(
                f"API key environment variable '{api_key_env}' is not set. "
                f"LLM calls will fail unless the key is provided."
            )

        if provider not in self.PROVIDER_MAP:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Supported: {list(self.PROVIDER_MAP.keys())}"
            )

        provider_class = self.PROVIDER_MAP[provider]
        self._client   = provider_class(model, temperature, max_tokens, api_key)

        logger.info(f"LLMClient initialized — provider={provider}, model={model}")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send prompt to configured LLM. Returns text response."""
        return self._client.complete(system_prompt, user_prompt)

    def complete_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        """Send prompt with automatic retry on transient errors."""
        return self._client.complete_with_retry(
            system_prompt, user_prompt, max_retries, retry_delay
        )
