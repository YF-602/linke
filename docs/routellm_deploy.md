部署 routellm/bert_gpt4_augmented 分类器并配置 RouteLLM 路由

此文档说明如何将 routellm/bert_gpt4_augmented（XML-RoBERTa 的 3 分类器）部署为一个 HTTP 服务，并如何配置 RouteLLM（仓库内的路由器）去调用该服务并将请求转发到目标模型。

我们提供两种部署方式：

- 轻量（推荐）: 使用 scripts/bert_service.py（FastAPI）在本机启动一个分类服务；
- 备选（云/HF）: 使用 Hugging Face Inference API（如果你有 HF token）。

环境准备

建议在虚拟环境（conda/venv）中执行：

1) 创建环境并安装依赖：
   conda create -n chuanhu-bert python=3.10 -y
   conda activate chuanhu-bert
   pip install fastapi uvicorn transformers torch

（注意：根据机器的 CUDA/CPU 环境，安装 torch 时选择适合的版本。）

使用脚本 scripts/bert_service.py

仓库中已提供 scripts/bert_service.py，它：

- 启动 FastAPI 服务；
- 加载 routellm/bert_gpt4_augmented（默认）；
- 提供 POST /predict 与 POST /v1/classifications 两个接口，输入 JSON { "inputs": "文本" }，返回 HF-inference 风格的 [ {"label": "LABEL_0", "score": 0.9}, ... ]。

启动示例（在项目根目录）：
HF_AUTH_TOKEN=你的_hf_token_if_needed uvicorn scripts.bert_service:app --host 0.0.0.0 --port 8800 --workers 1

启动成功后可以访问 http://<host>:8800/health 检查状态。

测试分类器（curl）示例：
curl -s -X POST "http://127.0.0.1:8800/predict" -H "Content-Type: application/json" -d '{"inputs":"请问如何写一封辞职信？"}'

把分类器当成 vllm/OpenAI-compatible 服务

RouteLLM 的实现会尝试以 HTTP endpoint 的形式调用分类器（ROUTELLM_BERT_URL 环境变量）。

scripts/bert_service.py 返回的 JSON 是兼容的；因此你可以把 ROUTELLM_BERT_URL 设置为 http://127.0.0.1:8800/predict 或 http://127.0.0.1:8800/v1/classifications。

如果你更希望在 vllm/OpenAI 样式的端点下运行分类器（例如 vllm 提供 OpenAI-compatible API），你可以搭建一个转换层把 /v1/models/.../predict 转为调用本脚本的 /predict，但并非必要。

RouteLLM 环境变量配置示例

下面示例假设你在本机 8800 端口运行分类器：

export ROUTELLM_BERT_URL="http://127.0.0.1:8800/predict"
export ROUTELLM_THRESHOLD=0.6
# mapping 格式为 JSON 字符串：{ "LABEL_0": "目标模型名A", ... }
export ROUTELLM_MAPPING='{"LABEL_0": "chatglm-6b", "LABEL_1": "GPT3.5 Turbo", "LABEL_2": "Llama-2-7B"}'
export ROUTELLM_FALLBACK_MODEL='GPT3.5 Turbo'

# 可选：HF token (仅当需要从 HF 下载 model 时)
export HF_AUTH_TOKEN="<your_hf_token>"

说明：
- ROUTELLM_MAPPING 中的目标模型名称应与 Chuanhu UI 中 MODELS 列表使用的显示名称一致（例如 GPT3.5 Turbo、chatglm-6b、Llama-2-7B 或你在 config.json 中增加的 extra_models 名称）。
- ROUTELLM_THRESHOLD 是得分阈值（0-1），低于阈值时会使用 ROUTELLM_FALLBACK_MODEL。

在 Chuanhu 中使用 RouteLLM

1. 在 Chuanhu 的 UI 下拉选择 RouteLLM（已在 MODELS 中注册）。
2. 在部署机器或进程中设置上面的 ROUTELLM_* 环境变量并重启 ChuanhuChatbot.py。
3. 向聊天输入问题，RouteLLM 会先调用分类器并根据映射转发到目标模型；你将看到转发后目标模型的回答。

备选：使用 Hugging Face Inference API

如果你没有本地推理资源，可以直接使用 HF Inference API：

1. 将 ROUTELLM_BERT_URL 设置为 HF Inference 的 endpoint（示例）：
export ROUTELLM_BERT_URL="https://api-inference.huggingface.co/models/routellm/bert_gpt4_augmented"
export HF_AUTH_TOKEN="<your_hf_token>"

2. HF Inference 返回结果多为 [{label, score}, ...]，RouteLLM 会直接解析。

监控与日志

- 查看 Chuanhu 运行日志（控制台）以观察路由器输出和目标模型加载错误；
- 查看 scripts/bert_service.py 的 uvicorn 控制台输出以调试分类器。

常见问题与建议

- 如果路由器选择目标模型失败，请检查 ROUTELLM_MAPPING 中的目标名称是否与 MODELS 列表一致；
- 若模型加载出现内存不足，请把分类器部署在单独机器或使用更小的模型版本，并在 RouteLLM 中把 ROUTELLM_BERT_URL 指向远程服务；
- 若担心安全，可把分类器服务绑定到本地 127.0.0.1，仅允许 Chuanhu 进程访问。

后续工作

如果你需要，我可以进一步：

- 帮你把 scripts/bert_service.py 打包成 Dockerfile 并给出 docker run 示例；
- 或者帮你在本项目内添加一个轻量的 wrapper，使分类器在 vllm 的 OpenAI-compatible 路径下可被直接调用（需要实现额外的 HTTP 路由）。
