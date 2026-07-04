import os

from openai import OpenAI, APIError, APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError


DEFAULT_MODEL = "deepseek-v4-pro"
MODEL_CHOICES = ["deepseek-v4-pro", "deepseek-v4-flash"]
DEFAULT_THINKING_TYPE = "enabled"
THINKING_TYPE_CHOICES = ["enabled", "disabled"]
DEFAULT_REASONING_EFFORT = "high"
REASONING_EFFORT_CHOICES = ["high", "max"]
API_TIMEOUT_DEFAULT = 90
API_TIMEOUT_SUPPORT = 25
API_TIMEOUT_FACT_CHECK = 12
API_TIMEOUT_LONG = 180


class DeepSeekService:
    def __init__(self):
        self._client = None
        self._api_key = None
        self._thinking_type = DEFAULT_THINKING_TYPE
        self._reasoning_effort = DEFAULT_REASONING_EFFORT

    def normalize_model(self, model):
        model = (model or DEFAULT_MODEL).strip()
        return model if model in MODEL_CHOICES else DEFAULT_MODEL

    def normalize_thinking_type(self, thinking_type):
        thinking_type = (thinking_type or DEFAULT_THINKING_TYPE).strip().lower()
        return thinking_type if thinking_type in THINKING_TYPE_CHOICES else DEFAULT_THINKING_TYPE

    def normalize_reasoning_effort(self, effort):
        effort = (effort or DEFAULT_REASONING_EFFORT).strip().lower()
        if effort in ("low", "medium", "high"):
            return "high"
        if effort in ("xhigh", "max"):
            return "max"
        return DEFAULT_REASONING_EFFORT

    def sync_api_settings(self, state):
        if not isinstance(state, dict):
            return DEFAULT_MODEL, DEFAULT_THINKING_TYPE, DEFAULT_REASONING_EFFORT
        model = self.normalize_model(state.get("model", DEFAULT_MODEL))
        thinking_type = self.normalize_thinking_type(state.get("thinking_type", DEFAULT_THINKING_TYPE))
        reasoning_effort = self.normalize_reasoning_effort(state.get("reasoning_effort", DEFAULT_REASONING_EFFORT))
        state["model"] = model
        state["thinking_type"] = thinking_type
        state["reasoning_effort"] = reasoning_effort
        self._thinking_type = thinking_type
        self._reasoning_effort = reasoning_effort
        return model, thinking_type, reasoning_effort

    def reset_client(self):
        self._client = None
        self._api_key = None

    def get_client(self, state):
        self.sync_api_settings(state)
        key = state.get("api_key", "").strip() if state else ""
        if not key:
            key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not key:
            return None

        if key != self._api_key or self._client is None:
            self._api_key = key
            self._client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=API_TIMEOUT_DEFAULT)
        return self._client

    def _message_content_text(self, message):
        content = getattr(message, "content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            content = "".join(parts)
        return str(content or "").strip()

    def call_deepseek(
        self,
        client,
        model,
        msgs,
        max_tok=1000,
        timeout=None,
        empty_retries=1,
        thinking_type=None,
        reasoning_effort=None,
    ):
        if client is None:
            return None, "请先在「启动 / API」里输入 DeepSeek API Key。"
        model = self.normalize_model(model)
        active_thinking_type = self.normalize_thinking_type(thinking_type or self._thinking_type)
        active_reasoning_effort = self.normalize_reasoning_effort(reasoning_effort or self._reasoning_effort)
        effective_timeout = int(timeout or API_TIMEOUT_DEFAULT)
        request_kwargs = {
            "model": model,
            "messages": msgs,
            "max_tokens": max_tok,
            "extra_body": {"thinking": {"type": active_thinking_type}},
        }
        if active_thinking_type == "enabled":
            request_kwargs["reasoning_effort"] = active_reasoning_effort
        else:
            request_kwargs["temperature"] = 0.7
        request_client = client.with_options(timeout=effective_timeout) if hasattr(client, "with_options") else client
        try:
            attempts = max(1, int(empty_retries or 1))
            for attempt in range(attempts):
                response = request_client.chat.completions.create(**request_kwargs)
                content = self._message_content_text(response.choices[0].message) if response.choices else ""
                if content:
                    return content, None
                if attempt < attempts - 1:
                    continue
            return None, "AI 返回空内容，请重试；如果反复出现，请降低回复长度预算或换一个模型。"
        except AuthenticationError:
            return None, "认证失败，请检查 DeepSeek API Key。"
        except RateLimitError:
            return None, "请求过于频繁，请稍后再试。"
        except APITimeoutError:
            return None, f"API 超过 {effective_timeout} 秒未返回，已中止本次请求。可以重试，或临时切换 v4Flash / 关闭思考模式。"
        except APIConnectionError as e:
            return None, f"API 连接失败：{e}"
        except APIError as e:
            return None, f"API 错误：{e}"
        except Exception as e:
            return None, f"错误：{e}"


deepseek_service = DeepSeekService()


def normalize_model(model):
    return deepseek_service.normalize_model(model)


def normalize_thinking_type(thinking_type):
    return deepseek_service.normalize_thinking_type(thinking_type)


def normalize_reasoning_effort(effort):
    return deepseek_service.normalize_reasoning_effort(effort)


def sync_api_settings(state):
    return deepseek_service.sync_api_settings(state)


def reset_client():
    return deepseek_service.reset_client()


def get_client(state):
    return deepseek_service.get_client(state)


def call_deepseek(client, model, msgs, max_tok=1000, timeout=None, empty_retries=1, thinking_type=None, reasoning_effort=None):
    return deepseek_service.call_deepseek(
        client,
        model,
        msgs,
        max_tok=max_tok,
        timeout=timeout,
        empty_retries=empty_retries,
        thinking_type=thinking_type,
        reasoning_effort=reasoning_effort,
    )

