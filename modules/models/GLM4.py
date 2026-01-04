from zai import ZhipuAiClient
from .base_model import BaseLLMModel
from ..utils import construct_system


class GLM4_Client(BaseLLMModel):
    def __init__(self, model_name, api_key, user_name="") -> None:
        super().__init__(
            model_name=model_name,   # e.g. "glm-4-7b-instruct"
            user=user_name,
            config={"api_key": api_key},
        )
        self.client = ZhipuAiClient(api_key=api_key)

    def _get_glm4_style_input(self):
        messages = [construct_system(self.system_prompt), *self.history]
        return messages

    def get_answer_at_once(self):
        messages = self._get_glm4_style_input()
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            # thinking={"type": "enabled"},  # 开启深度思考
            temperature=self.temperature,
            max_tokens=self.max_generation_token,
        )
        return resp.choices[0].message.content, resp.usage.total_tokens

    def get_answer_stream_iter(self):
        messages = self._get_glm4_style_input()
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            # thinking={"type": "enabled"},
            stream=True,
            temperature=self.temperature,
            max_tokens=self.max_generation_token,
        )

        partial_text = ""
        for chunk in stream:
            # 取 delta.content，如果是 None 就用空字符串
            content = getattr(chunk.choices[0].delta, "content", "") or ""
            # 如果 content 是 dict（ZhipuAiClient 可能返回 dict），尝试取 text
            if isinstance(content, dict):
                content = content.get("text", "")

            partial_text += content
            yield partial_text
