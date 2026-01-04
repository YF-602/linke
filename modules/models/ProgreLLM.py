from __future__ import annotations

import json
import logging
import os
from urllib.parse import urlparse

import requests
from html import escape
import re

from .base_model import BaseLLMModel
from ..presets import i18n


class ProgreLLM_Client(BaseLLMModel):
    """Virtual model that runs a large-model draft step followed by a
    small-model completion step (progressive reasoning).

    The progressive endpoints/config live in the model metadata under
    `metadata['progressive']` with keys `large_endpoint`, `small_endpoint`,
    `large_api_key`, `small_api_key`. Each endpoint supports the same
    formats as `ROUTELLM_MAPPING` (e.g. `http://host:port/v1/chat/completions|model-name`,
    or an http URL, or a local model name).
    """

    def __init__(self, model_name: str, user_name: str = "") -> None:
        super().__init__(model_name=model_name, user=user_name)
        self.progressive = self.metadata.get("progressive", {}) or {}

    def _is_url(self, s: str) -> bool:
        return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

    def _is_status_text(self, s: str) -> bool:
        if not isinstance(s, str):
            return False
        # common status / token messages to ignore
        pat = r"(Token Count:|total cost:|Tokens per second|Token per second|Tokens per sec|开始生成回答|正在实时传输)"
        if re.search(pat, s, re.IGNORECASE):
            return True
        # also ignore very short status-like messages
        if len(s.strip()) <= 4 and not any(c.isalpha() for c in s):
            return True
        return False

    def _call_http_once(self, url: str, prompt_text: str, api_key: str = None) -> str:
        try:
            headers = {"Content-Type": "application/json"}
            # allow passing api key for HF/vllm services
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            endpoint = url
            specified_model = None
            if isinstance(url, str) and "|" in url:
                parts = url.split("|", 1)
                endpoint = parts[0].strip()
                specified_model = parts[1].strip() or None

            parsed = urlparse(endpoint)
            vllm_url = f"{parsed.scheme}://{parsed.netloc}"

            vllm_model_name = specified_model
            if vllm_model_name is None:
                try:
                    resp = requests.get(f"{vllm_url}/v1/models", headers=headers, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
                    vllm_model_name = models[0] if models else None
                except Exception:
                    vllm_model_name = None

            payload = {
                "model": vllm_model_name,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": 2048,
            }
            r = requests.post(endpoint, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "choices" in data:
                first = data["choices"][0]
                if "message" in first and "content" in first["message"]:
                    return first["message"]["content"]
                if "text" in first:
                    return first["text"]
            if isinstance(data, str):
                return data
            return json.dumps(data)
        except Exception:
            logging.exception("ProgreLLM: HTTP 调用失败")
            raise

    def _call_http_stream(self, url: str, prompt_text: str, api_key: str = None):
        try:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            endpoint = url
            specified_model = None
            if isinstance(url, str) and "|" in url:
                parts = url.split("|", 1)
                endpoint = parts[0].strip()
                specified_model = parts[1].strip() or None

            parsed = urlparse(endpoint)
            vllm_url = f"{parsed.scheme}://{parsed.netloc}"

            vllm_model_name = specified_model
            if vllm_model_name is None:
                try:
                    resp = requests.get(f"{vllm_url}/v1/models", headers=headers, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
                    vllm_model_name = models[0] if models else None
                except Exception:
                    vllm_model_name = None

            payload = {
                "model": vllm_model_name,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": 2048,
                "stream": True,
            }

            r = requests.post(endpoint, json=payload, headers=headers, timeout=120, stream=True)
            r.raise_for_status()

            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                if line == "[DONE]":
                    break
                # Try parse JSON prefix like RouteLLM
                remaining = line
                parsed_any = False
                while remaining:
                    remaining = remaining.lstrip()
                    if not remaining:
                        break
                    if not remaining.startswith("{"):
                        yield remaining
                        break
                    found = False
                    for i in range(1, len(remaining) + 1):
                        prefix = remaining[:i]
                        try:
                            j = json.loads(prefix)
                            parsed_any = True
                            found = True
                            remaining = remaining[i:]
                            chunk = ""
                            if isinstance(j, dict):
                                if "choices" in j and isinstance(j["choices"], list) and len(j["choices"]) > 0:
                                    ch = j["choices"][0]
                                    if isinstance(ch, dict):
                                        if "delta" in ch and isinstance(ch["delta"], dict):
                                            chunk = ch["delta"].get("content") or ch["delta"].get("text") or ""
                                        elif "text" in ch:
                                            chunk = ch.get("text", "")
                                        elif "message" in ch and isinstance(ch["message"], dict):
                                            chunk = ch["message"].get("content", "")
                                elif "text" in j:
                                    chunk = j.get("text", "")
                                elif "message" in j and isinstance(j["message"], dict):
                                    chunk = j["message"].get("content", "")
                            if chunk:
                                yield chunk
                            break
                        except Exception:
                            continue
                    if not found:
                        if not parsed_any:
                            yield line
                        else:
                            if remaining:
                                yield remaining
                        break
            return
        except Exception:
            logging.exception("ProgreLLM: HTTP 流式调用失败")
            raise

    def _call_local_model_once(self, model_spec: str, prompt_text: str):
        # Use get_model factory to instantiate and call predict once
        try:
            from .models import get_model as _get_model
        except Exception:
            from modules.models.models import get_model as _get_model

        try:
            target_model, msg, _, _, access_key, _, _, stream_flag = _get_model(model_name=model_spec, access_key=None, user_name=self.user_name, original_model=None)
        except Exception:
            logging.exception("ProgreLLM: 获取本地模型失败")
            return ""

        # Collect responses yielded by the model (whether streaming or at-once)
        replies = []
        try:
                for out_chatbot, out_status in target_model.predict(prompt_text, []):
                    # Prefer assistant content from out_chatbot when present
                    appended = False
                    if isinstance(out_chatbot, list) and len(out_chatbot) > 0:
                        try:
                            val = out_chatbot[-1][1] if len(out_chatbot[-1]) > 1 else ""
                            if isinstance(val, str) and val.strip():
                                replies.append(val)
                                appended = True
                        except Exception:
                            pass
                    if not appended and isinstance(out_status, str) and out_status.strip() and not self._is_status_text(out_status):
                        replies.append(out_status)
        except Exception:
            logging.exception("ProgreLLM: 本地模型调用失败")

        # Prefer the last captured reply; otherwise try model.history
        if replies:
            return replies[-1]
        try:
            if hasattr(target_model, "history") and len(target_model.history) > 0:
                return target_model.history[-1]["content"]
        except Exception:
            pass
        return ""

    def predict(self, inputs, chatbot, use_websearch=False, files=None, reply_language="中文", should_check_token_count=True):
        status_text = i18n("开始渐进式推理（大模型草稿 -> 小模型补全）……")
        # reuse prepare_inputs from base class to get fake_inputs
        limited_context, fake_inputs, display_append, real_inputs, chatbot = self.prepare_inputs(
            real_inputs=inputs, use_websearch=use_websearch, files=files, reply_language=reply_language, chatbot=chatbot
        )

        if should_check_token_count:
            # show status without injecting an empty assistant entry
            yield chatbot, status_text

        if len(fake_inputs.strip()) == 0:
            yield chatbot, i18n("输入为空，无法生成")
            return

        # Build the large model prompt with the simplification instruction
        simplification_prefix = "Please answer the question with the extreme simplification at the grammatical level, regardless of the grammer error as long as the answer is comprehensible, e.g., deleting all the 'the' et al. Now, please answer question:"
        large_prompt = simplification_prefix + "\n" + fake_inputs

        # Determine large and small endpoints/specs
        prog = self.progressive
        large_spec = prog.get("large_endpoint") or prog.get("large") or ""
        small_spec = prog.get("small_endpoint") or prog.get("small") or ""
        large_api_key = prog.get("large_api_key") or None
        small_api_key = prog.get("small_api_key") or None

        # Fallback: if specs empty, return error message
        if not large_spec or not small_spec:
            yield chatbot + [(fake_inputs, "")], i18n("渐进式推理未配置大/小模型端点，请在模型设置中填写 progressive 配置")
            return

        # Call large model, prefer streaming so the user can see draft
        # generation. Append the single assistant message only when the
        # first chunk of draft is available to avoid showing an empty
        # reply.
        # Ensure the user's message is appended so the UI shows the
        # original question instead of immediately showing the assistant
        # entry.
        if not any(isinstance(c, (list, tuple)) and c[0] == fake_inputs for c in chatbot):
            chatbot.append((fake_inputs, ""))
        yield chatbot, i18n("正在调用大模型生成草稿……")
        draft = ""
        draft_accum = ""
        assistant_appended = False
        # cache for a local large-model instance to avoid re-loading same model
        large_local_model_instance = None
        try:
            if self._is_url(large_spec) or ("|" in large_spec and large_spec.split("|",1)[0].startswith("http")):
                # try streaming HTTP for draft
                try:
                    for chunk in self._call_http_stream(large_spec, large_prompt, api_key=large_api_key):
                        draft_accum += chunk
                        details_html_live = f'<details><summary>草稿（大模型）</summary><pre style="white-space:pre-wrap;background:#f6f7f8;color:#111;padding:8px;border-radius:6px;">{escape(draft_accum)}</pre></details>\n\n'
                        if not assistant_appended:
                            chatbot[-1] = (chatbot[-1][0], details_html_live + "")
                            assistant_appended = True
                        else:
                            chatbot[-1] = (chatbot[-1][0], details_html_live + "")
                        yield chatbot, i18n("正在从大模型流式生成草稿……")
                    draft = draft_accum
                except Exception:
                    # fallback to single-call
                    draft = self._call_http_once(large_spec, large_prompt, api_key=large_api_key)
                    details_html = f'<details><summary>草稿（大模型）</summary><pre style="white-space:pre-wrap;background:#f6f7f8!important;color:#111!important;padding:8px;border-radius:6px;">{escape(draft)}</pre></details>\n\n'
                    chatbot[-1] = (chatbot[-1][0], details_html + "")
                    assistant_appended = True
            else:
                # local model: try streaming predict
                try:
                    from modules.models.models import get_model as _get_model
                except Exception:
                    from .models import get_model as _get_model

                try:
                    local_model, msg, _, _, access_key, _, _, stream_flag = _get_model(model_name=large_spec, access_key=None, user_name=self.user_name, original_model=None)
                    # cache instance to reuse for small model if specs match
                    large_local_model_instance = local_model
                except Exception:
                    logging.exception("ProgreLLM: 获取本地大模型失败")
                    draft = self._call_local_model_once(large_spec, large_prompt)
                    details_html = f'<details><summary>草稿（大模型）</summary><pre style="white-space:pre-wrap;background:#f6f7f8!important;color:#111!important;padding:8px;border-radius:6px;">{escape(draft)}</pre></details>\n\n'
                    chatbot[-1] = (chatbot[-1][0], details_html + "")
                    assistant_appended = True
                    draft = draft
                    # continue outer flow
                else:
                    try:
                        # Iterate through the model.predict generator. Do not pass an unsupported 'stream' kw.
                        for out_chatbot, out_status in local_model.predict(large_prompt, []):
                            # Prefer assistant content from out_chatbot when present
                            chunk_text = ""
                            if isinstance(out_chatbot, list) and len(out_chatbot) > 0:
                                try:
                                    val = out_chatbot[-1][1] if len(out_chatbot[-1]) > 1 else ""
                                    if isinstance(val, str) and val.strip():
                                        chunk_text = val
                                except Exception:
                                    chunk_text = ""
                            if not chunk_text and isinstance(out_status, str) and out_status.strip() and not self._is_status_text(out_status):
                                chunk_text = out_status
                            draft_accum += chunk_text
                            details_html_live = f'<details><summary>草稿（大模型）</summary><pre style="white-space:pre-wrap;background:#f6f7f8!important;color:#111!important;padding:8px;border-radius:6px;">{escape(draft_accum)}</pre></details>\n\n'
                            if not assistant_appended:
                                chatbot[-1] = (chatbot[-1][0], details_html_live + "")
                                assistant_appended = True
                            else:
                                chatbot[-1] = (chatbot[-1][0], details_html_live + "")
                            yield chatbot, i18n("正在从大模型流式生成草稿……")
                        draft = draft_accum
                    except Exception:
                        logging.exception("ProgreLLM: 本地大模型流式生成失败，回退为一次性调用")
                        draft = self._call_local_model_once(large_spec, large_prompt)
                        details_html = f'<details><summary>草稿（大模型）</summary><pre style="white-space:pre-wrap;background:#f6f7f8!important;color:#111!important;padding:8px;border-radius:6px;">{escape(draft)}</pre></details>\n\n'
                        chatbot[-1] = (chatbot[-1][0], details_html + "")
                        assistant_appended = True
        except Exception as e:
            logging.exception("ProgreLLM: 大模型生成失败")
            yield chatbot + [(fake_inputs, "")], i18n("大模型生成失败，错误：") + str(e)
            return

        # final draft available (details are either already appended or
        # will be updated). Ensure `details_html` holds the final draft
        # block for downstream updates, then notify status and proceed.
        details_html = f'<details><summary>草稿（大模型）</summary><pre style="white-space:pre-wrap;background:#f6f7f8!important;color:#111!important;padding:8px;border-radius:6px;">{escape(draft)}</pre></details>\n\n'
        yield chatbot, i18n("已获得大模型草稿，正在调用小模型补全……")

        # Build combined prompt for small model. Instruct the small model to
        # strictly align its final answer to the original user's question.
        combined_prompt = (
            draft
            + "\n\n"
            + fake_inputs
            + "\nPlease ensure the final answer directly and concisely answers the user's original question; do not introduce unrelated information."
            + "\nComplete the answer content based on the draft and the questions"
        )

        # Call small model and stream its output to UI
        try:
            if self._is_url(small_spec) or ("|" in small_spec and small_spec.split("|",1)[0].startswith("http")):
                # stream from HTTP endpoint if possible
                try:
                    stream_gen = self._call_http_stream(small_spec, combined_prompt, api_key=small_api_key)
                    # place final-answer placeholder into the existing user entry
                    if not assistant_appended:
                        chatbot[-1] = (chatbot[-1][0], details_html + "")
                        assistant_appended = True
                    else:
                        chatbot[-1] = (chatbot[-1][0], details_html + "")
                    accum = ""
                    for chunk in stream_gen:
                        accum += chunk
                        chatbot[-1] = (chatbot[-1][0], details_html + escape(accum))
                        yield chatbot, i18n("正在从小模型流式接收回答……")
                    # final
                    chatbot[-1] = (chatbot[-1][0], details_html + escape(accum))
                    yield chatbot, i18n("小模型补全完成")
                except Exception:
                    # fallback to single-call
                    reply = self._call_http_once(small_spec, combined_prompt, api_key=small_api_key)
                    if not assistant_appended:
                        chatbot[-1] = (chatbot[-1][0], details_html + escape(reply))
                        assistant_appended = True
                    else:
                        chatbot[-1] = (chatbot[-1][0], details_html + escape(reply))
                    yield chatbot, i18n("小模型补全完成（一次性返回）")
            else:
                # local model: delegate to model.predict and stream its yields
                try:
                    from modules.models.models import get_model as _get_model
                except Exception:
                    from .models import get_model as _get_model

                # If small_spec equals large_spec and we have a cached instance, pass it as original_model
                try:
                    orig = large_local_model_instance if (large_local_model_instance is not None and small_spec == large_spec) else None
                    target_model, msg, _, _, access_key, _, _, stream_flag = _get_model(model_name=small_spec, access_key=None, user_name=self.user_name, original_model=orig)
                except Exception:
                    logging.exception("ProgreLLM: 获取本地小模型失败")
                    reply = self._call_local_model_once(small_spec, combined_prompt)
                    if not assistant_appended:
                        chatbot[-1] = (chatbot[-1][0], details_html + escape(reply))
                        assistant_appended = True
                    else:
                        chatbot[-1] = (chatbot[-1][0], details_html + escape(reply))
                    yield chatbot, i18n("小模型补全完成（一次性返回）")
                else:
                    # create placeholder with embedded draft in the existing user entry
                    if not assistant_appended:
                        chatbot[-1] = (chatbot[-1][0], details_html + "")
                        assistant_appended = True
                    else:
                        chatbot[-1] = (chatbot[-1][0], details_html + "")
                    accum = ""
                    try:
                        # Iterate the predict generator without passing unsupported kwargs
                        for out_chatbot, out_status in target_model.predict(combined_prompt, []):
                            # Prefer assistant content from out_chatbot when present
                            if isinstance(out_chatbot, list) and len(out_chatbot) > 0:
                                try:
                                    val = out_chatbot[-1][1] if len(out_chatbot[-1]) > 1 else ""
                                    if isinstance(val, str) and val.strip():
                                        accum = val
                                except Exception:
                                    pass
                            if (not accum or not isinstance(accum, str) or not accum.strip()) and isinstance(out_status, str) and out_status.strip() and not self._is_status_text(out_status):
                                accum = out_status
                            chatbot[-1] = (chatbot[-1][0], details_html + escape(accum))
                            yield chatbot, i18n("正在从小模型流式接收回答……")
                    except Exception:
                        # fallback to once
                        logging.exception("ProgreLLM: 本地小模型流式生成失败，回退为一次性调用")
                        reply = self._call_local_model_once(small_spec, combined_prompt)
                        if not assistant_appended:
                            chatbot[-1] = (chatbot[-1][0], details_html + escape(reply))
                            assistant_appended = True
                        else:
                            chatbot[-1] = (chatbot[-1][0], details_html + escape(reply))
                        yield chatbot, i18n("小模型补全完成（一次性返回）")
        except Exception:
            logging.exception("ProgreLLM: 小模型调用失败")
            yield chatbot, i18n("小模型调用失败，请查看日志")
