from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9000/v1",
    api_key="EMPTY"
)

resp = client.chat.completions.create(
    model="qwen3-4b-instruct",
    messages=[
        {"role": "user", "content": "你好"}
    ]
)

print(resp.choices[0].message.content)
