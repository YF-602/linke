import os
import threading
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

from .base_model import BaseLLMModel
from ..presets import MODEL_METADATA


class Qwen3_Client(BaseLLMModel):
    """
    Qwen3 专用 Client
    - 基于 Transformers generate()
    - 使用 tokenizer.apply_chat_template()
    - 支持流式 / 非流式
    """

    def __init__(self, model_name, user_name="") -> None:
        super().__init__(model_name=model_name, user=user_name)

        # 1. 解析模型来源（本地优先，其次 HF repo）
        model_source = None
        if os.path.exists("models"):
            model_dirs = os.listdir("models")
            if model_name in model_dirs:
                model_source = os.path.join("models", model_name)

        if model_source is None:
            try:
                model_source = MODEL_METADATA[model_name]["repo_id"]
            except KeyError:
                model_source = model_name

        logging.info(f"[Qwen3] loading model from: {model_source}")

        # 2. 加载 tokenizer / model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=True,
            resume_download=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            device_map="cuda",
            trust_remote_code=True,
            resume_download=True,
        ).eval()

    # ------------------------------------------------------------------
    # Prompt 构造（Qwen3 官方方式）
    # ------------------------------------------------------------------
    def _build_prompt(self) -> str:
        """
        使用 Qwen3 / Transformers 官方 chat template
        """
        messages = []
        for item in self.history:
            messages.append(
                {
                    "role": item["role"],
                    "content": item["content"],
                }
            )

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return prompt

    # ------------------------------------------------------------------
    # 非流式生成
    # ------------------------------------------------------------------
    def get_answer_at_once(self):
        prompt = self._build_prompt()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 只解码新生成部分
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return response, len(response)

    # ------------------------------------------------------------------
    # 流式生成（真正 streaming）
    # ------------------------------------------------------------------
    def get_answer_stream_iter(self):
        prompt = self._build_prompt()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        thread = threading.Thread(
            target=self.model.generate,
            kwargs=generation_kwargs,
        )
        thread.start()

        full_text = ""   # ⭐ 关键：自己维护完整文本

        for new_text in streamer:
            full_text += new_text
            yield full_text   # ✅ 每次 yield「完整回答」
