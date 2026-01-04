下面介绍如何使用模型路由：

首先部署vllm
1. 首先创建conda环境vllm：`conda create -n vllm python=3.10`
2. 之后在该环境安装：`pip install vllm`
3. 命令行输入
```
CUDA_VISIBLE_DEVICES=1 \
python -m vllm.entrypoints.openai.api_server \
--config ./vllm_config.yml
```
至此完成vllm部署
vllm设置在vllm_config.yml

之后部署bert-router
1. 首先创建conda环境bert-router：`conda create -n bert-router python=3.10`
2. 之后在该环境安装：`pip install fastapi uvicorn transformers torch`
3. 在终端输入`uvicorn scripts.bert_service:app --host 0.0.0.0 --port 8800 --workers 1`
至此完成bert-router的部署，通过调用FastAPI提供的http接口（这里设置为了8800）进行分类路由

最后，启动./ChuanhuChatbot.py，在下拉栏选择RouteLLM即可
Router设置在presets.py