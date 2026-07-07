import json

from src.utils.logger import get_logger

log = get_logger("cloud.llm")


class LLMAdapter:
    def __init__(self, config: dict):
        self._config = config
        llm_cfg = config.get("llm", {})
        self._provider = llm_cfg.get("provider", "anthropic").lower()
        self._model = llm_cfg.get("model", None)
        self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict],
        system: str = "",
        model: str = None,
        max_tokens: int = 2048,
    ) -> str:
        resolved_model = model or self._model
        log.info(f"LLM complete — provider={self._provider}, model={resolved_model}")

        if self._provider == "anthropic":
            return self._complete_anthropic(messages, system, resolved_model, max_tokens)
        elif self._provider == "bedrock":
            return self._complete_bedrock(messages, system, resolved_model, max_tokens)
        elif self._provider == "azure":
            return self._complete_azure(messages, system, resolved_model, max_tokens)
        else:
            raise ValueError(
                f"Unknown LLM provider '{self._provider}'. "
                "Valid values: 'anthropic' | 'bedrock' | 'azure'"
            )

    def get_model_info(self) -> dict:
        llm_cfg = self._config.get("llm", {})
        info = {"provider": self._provider, "model": self._model}

        if self._provider == "anthropic":
            info["api_key_set"] = bool(llm_cfg.get("api_key"))
        elif self._provider == "bedrock":
            info["region"] = llm_cfg.get("region", "us-east-1")
            info["model_id"] = llm_cfg.get("model_id", "anthropic.claude-sonnet-4-5")
        elif self._provider == "azure":
            info["endpoint"] = llm_cfg.get("endpoint", "")
            info["deployment"] = llm_cfg.get("deployment", "")
            info["api_version"] = "2024-02-01"

        return info

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _complete_anthropic(
        self,
        messages: list[dict],
        system: str,
        model: str,
        max_tokens: int,
    ) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is not installed. "
                "Install it with: pip install anthropic"
            )

        if self._client is None:
            llm_cfg = self._config.get("llm", {})
            api_key = llm_cfg.get("api_key")
            self._client = anthropic.Anthropic(api_key=api_key)

        kwargs = dict(
            model=model or "claude-sonnet-4-6",
            max_tokens=max_tokens,
            messages=messages,
        )
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def _complete_bedrock(
        self,
        messages: list[dict],
        system: str,
        model: str,
        max_tokens: int,
    ) -> str:
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is not installed. "
                "Install it with: pip install boto3"
            )

        llm_cfg = self._config.get("llm", {})
        region = llm_cfg.get("region", "us-east-1")
        model_id = model or llm_cfg.get("model_id", "anthropic.claude-sonnet-4-5")

        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=region)

        payload: dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            payload["system"] = system

        response = self._client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        body = json.loads(response["body"].read())
        return body["content"][0]["text"]

    def _complete_azure(
        self,
        messages: list[dict],
        system: str,
        model: str,
        max_tokens: int,
    ) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package is not installed. "
                "Install it with: pip install openai"
            )

        llm_cfg = self._config.get("llm", {})

        if self._client is None:
            self._client = openai.AzureOpenAI(
                azure_endpoint=llm_cfg.get("endpoint", ""),
                api_key=llm_cfg.get("api_key", ""),
                api_version="2024-02-01",
            )

        deployment = model or llm_cfg.get("deployment", "")
        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(messages)

        response = self._client.chat.completions.create(
            model=deployment,
            messages=openai_messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


def get_llm(config: dict) -> LLMAdapter:
    return LLMAdapter(config)
