from __future__ import annotations

import json
import logging
import os
from typing import Dict, List

import requests

from .base_model import BaseLLMModel

from ..presets import i18n


class RouteLLM_Client(BaseLLMModel):
    """A simple router model that classifies an input prompt (using a small
    classifier such as the routellm/bert_gpt4_augmented) and then dispatches
    the request to a target model based on the predicted class.

    The classifier can be called via an HTTP endpoint (set `ROUTELLM_BERT_URL`)
    — useful when the classifier is served as a vllm/openai-compatible service —
    or will fall back to a local Hugging Face `text-classification` pipeline
    (requires `transformers`).
    """

    def __init__(self, model_name: str, user_name: str = "") -> None:
        super().__init__(model_name=model_name, user=user_name)
        # HTTP endpoint for classifier service (optional)
        self.bert_url = os.environ.get("ROUTELLM_BERT_URL", "")
        # threshold for deciding to use top prediction
        self.threshold = float(os.environ.get("ROUTELLM_THRESHOLD", 0.6))
        # mapping from classifier label to target model name
        mapping_env = os.environ.get("ROUTELLM_MAPPING", "{}")
        try:
            self.mapping: Dict[str, str] = json.loads(mapping_env) if mapping_env else {}
        except Exception:
            logging.exception("无法解析 ROUTELLM_MAPPING，使用空映射")
            self.mapping = {}

        # default fallback model if no label passes threshold
        self.fallback_model = os.environ.get("ROUTELLM_FALLBACK_MODEL", "GPT3.5 Turbo")

        # local classifier fallback
        self._local_classifier = None
        if not self.bert_url:
            try:
                from transformers import pipeline

                self._local_classifier = pipeline(
                    "text-classification",
                    model="routellm/bert_gpt4_augmented",
                    return_all_scores=True,
                )
                logging.info("RouteLLM: 已加载本地 transformers classifier")
            except Exception as e:
                logging.warning(f"RouteLLM: 无法加载本地 classifier: {e}")

    def _classify_via_http(self, text: str) -> List[Dict]:
        """Call a HTTP classifier endpoint. Expected response formats supported:
        - HF Inference style: a list of {label, score}
        - [{'label': 'LABEL_1', 'score': 0.9}, ...]
        - or {'labels': [...], 'scores': [...]} style
        Returns a list of dicts {'label': str, 'score': float}
        """
        try:
            headers = {"Content-Type": "application/json"}
            # If HF token provided in env, attach Authorization
            hf_token = os.environ.get("HF_AUTH_TOKEN")
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
            r = requests.post(self.bert_url, json={"inputs": text}, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            # HF inference returns a list of {label,score}
            if isinstance(data, list):
                # e.g. [{'label':'LABEL_0', 'score':0.9}, ...] or [[...]]
                # normalize
                result = []
                if len(data) > 0 and isinstance(data[0], list):
                    # sometimes returned as [[{...}, {...}]]
                    data = data[0]
                for item in data:
                    if isinstance(item, dict) and "label" in item and "score" in item:
                        result.append({"label": item["label"], "score": float(item["score"])})
                return result
            if isinstance(data, dict):
                # try labels/scores
                if "labels" in data and "scores" in data:
                    return [{"label": l, "score": float(s)} for l, s in zip(data["labels"], data["scores"]) if l is not None]
            logging.warning("RouteLLM: 未知的HTTP分类器响应格式，返回空列表")
            return []
        except Exception:
            logging.exception("RouteLLM: HTTP 分类器调用失败")
            return []

    def _classify_local(self, text: str) -> List[Dict]:
        if self._local_classifier is None:
            logging.warning("RouteLLM: 本地 classifier 未就绪")
            return []
        try:
            res = self._local_classifier(text)
            # `res` is a list of lists (return_all_scores=True) -> take first
            if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                res = res[0]
            normalized = [{"label": r.get("label", ""), "score": float(r.get("score", 0.0))} for r in res]
            return normalized
        except Exception:
            logging.exception("RouteLLM: 本地分类失败")
            return []

    def classify(self, text: str) -> List[Dict]:
        if self.bert_url:
            out = self._classify_via_http(text)
            if out:
                return out
            # fallback to local
        return self._classify_local(text)

    def predict(
        self,
        inputs,
        chatbot,
        use_websearch=False,
        files=None,
        reply_language="中文",
        should_check_token_count=True,
    ):
        # route for a single user query (non-batched)
        if isinstance(inputs, list):
            prompt = inputs[0].get("text", "")
        else:
            prompt = inputs

        # Inform UI we are routing
        status_text = i18n("正在进行模型路由判断……")
        yield chatbot + [(prompt, "")], status_text

        # classify
        scores = self.classify(prompt)
        if not scores:
            status_text = i18n("路由器无法得到分类结果，使用回退模型")
            target_model_name = self.fallback_model
        else:
            # 打印所有分类结果
            for s in scores:
                label = s.get("label")
                score = s.get("score", 0.0)
                logging.info(f"RouteLLM: 分类结果: {label} (score {score:.4f})")
            # choose highest
            best = max(scores, key=lambda x: x.get("score", 0.0))
            label = best.get("label")
            score = best.get("score", 0.0)

            logging.info(f"RouteLLM: 最终分类结果: {label} (score {score:.4f})")

            mapped = self.mapping.get(label, None)
            if score >= self.threshold and mapped:
                target_model_name = mapped
                status_text = i18n("路由结果：{label} (概率 {score:.2f})，将转发到 {model}").format(label=label, score=score, model=target_model_name)
            else:
                target_model_name = self.fallback_model
                status_text = i18n("没有达到阈值，使用回退模型 {model}").format(model=target_model_name)

        yield chatbot + [(prompt, "")], status_text

        # avoid recursive routing
        if target_model_name == self.model_name:
            yield chatbot + [(prompt, "")], i18n("路由器选择了自身，停止以避免死循环")
            return

        # If mapping points to an HTTP endpoint (e.g. a vllm/OpenAI-compatible API), call it directly
        def _is_url(s: str) -> bool:
            return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

        def _call_http_target(url: str, prompt_text: str) -> str:
            try:
                headers = {"Content-Type": "application/json"}
                hf_token = os.environ.get("HF_AUTH_TOKEN")
                if hf_token:
                    headers["Authorization"] = f"Bearer {hf_token}"
                # Try OpenAI-compatible chat completions payload first
                payload = {
                    "model": os.environ.get("ROUTELLM_HTTP_MODEL_NAME", "routellm-proxy"),
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": 1024,
                }
                r = requests.post(url, json=payload, headers=headers, timeout=30)
                r.raise_for_status()
                data = r.json()
                # OpenAI-style response
                if isinstance(data, dict) and "choices" in data:
                    # pick first choice
                    first = data["choices"][0]
                    # chat completion
                    if "message" in first and "content" in first["message"]:
                        return first["message"]["content"]
                    # completion style
                    if "text" in first:
                        return first["text"]
                # if returned directly as text or list
                if isinstance(data, str):
                    return data
                # fallback: try HF-inference style list
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "label" in data[0]:
                    # not a generative reply; join labels
                    return ", ".join([f"{it.get('label')}({it.get('score'):.2f})" for it in data])
                return json.dumps(data)
            except Exception:
                logging.exception("RouteLLM: 调用 HTTP 目标失败")
                raise

        # If target is a URL, call it directly
        if _is_url(target_model_name):
            try:
                reply = _call_http_target(target_model_name, prompt)
                chatbot.append((prompt, reply))
                yield chatbot, i18n("已从 HTTP 目标获得回答")
                return
            except Exception:
                logging.warning("RouteLLM: 调用映射的 HTTP 目标失败，尝试使用回退模型")
                # fall through to use get_model with fallback

        # lazily import factory to avoid circular import on module load
        try:
            from modules.models.models import get_model
        except Exception:
            logging.exception("RouteLLM: 无法导入 get_model")
            yield chatbot + [(prompt, "")], i18n("内部错误：无法加载目标模型")
            return

        # Try to get target model via factory; if it fails due to network, try fallback
        try:
            target_model, msg, placeholder_update, dropdown_update, access_key, presudo_key, modelDescription, stream = get_model(
                model_name=target_model_name, access_key=self.api_key, user_name=self.user_name, original_model=None
            )
        except Exception:
            logging.exception("RouteLLM: 获取目标模型失败，尝试回退模型")
            # If fallback differs, try fallback
            if target_model_name != self.fallback_model:
                try:
                    target_model, msg, placeholder_update, dropdown_update, access_key, presudo_key, modelDescription, stream = get_model(
                        model_name=self.fallback_model, access_key=self.api_key, user_name=self.user_name, original_model=None
                    )
                except Exception:
                    logging.exception("RouteLLM: 回退模型也无法加载")
                    yield chatbot + [(prompt, "")], i18n("获取目标模型失败，请查看日志")
                    return
            else:
                yield chatbot + [(prompt, "")], i18n("获取目标模型失败，请查看日志")
                return

        # Delegate the actual prediction to the chosen model and forward its yields
        try:
            for out_chatbot, out_status in target_model.predict(
                inputs, chatbot, use_websearch=use_websearch, files=files, reply_language=reply_language
            ):
                yield out_chatbot, out_status
        except Exception:
            logging.exception("RouteLLM: 目标模型预测失败")
            yield chatbot + [(prompt, "")], i18n("目标模型在推理时失败，请查看日志")
