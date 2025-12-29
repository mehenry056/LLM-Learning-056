import os
from openai import OpenAI
client = OpenAI(
    api_key="sk-a6dafdc592cb444e894133d7d68220d0",
    # 以下是北京地域base_url，如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen3-max",  # 支持联网的高级模型
    messages=[
        {"role": "user", "content": "2025年12月29日，北京的实时天气和新闻是什么？"}
    ],
    temperature=0.7,
    stream=False,  # 改为 True 可流式输出
    extra_body={"enable_search": True}  # ← 正确位置！
)

print(completion.choices[0].message.content)