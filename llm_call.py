import gradio as gr
import os
import statics
from openai import OpenAI

def chat_with_llm(message, history, provider, model, temperature):
    if not message or not message.strip():
        yield "", history
        return

    # 1. 这里的 history 现在是一个 list of dicts: [{'role': 'user', 'content': '...'}, ...]
    provider = provider.lower()
    api_key_env = statics.API_KEY_ENV.get(provider)

    api_key = os.getenv(api_key_env)

    if not api_key:
        error_msg = f"⚠️ 未检测到 {provider.upper()} 的 API Key！"
        # 兼容新格式：添加用户消息和错误提示
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        yield "", history
        return

    try:
        # 客户端初始化保持不变
        if provider == "dashscope":
            client = OpenAI(api_key=api_key, base_url = statics.BASE_URL_MAP["dashscope"])
            extra_body = {"enable_search": True}
        elif provider == "groq":
            client = OpenAI(api_key=api_key, base_url=statics.BASE_URL_MAP["groq"])
            extra_body = {}
        elif provider == "gemini":
            client = OpenAI(api_key=api_key, base_url = statics.BASE_URL_MAP["gemini"])
            extra_body = {}
        elif provider == "grok":
            client = OpenAI(api_key=api_key, base_url= statics.BASE_URL_MAP["grok"])
            extra_body = {}
        else:
            client = OpenAI(api_key=api_key)
            extra_body = {}

        # 2. 构造发送给 API 的 messages（现在 history 已经是这个格式了，可以直接追加）
        messages = history + [{"role": "user", "content": message}]

        # 3. 更新 UI：先显示用户消息，并预留一个空的助手回复位
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        yield "", history

        full_response = ""
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
            extra_body=extra_body
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                # 更新最后一条助手消息的内容
                history[-1]["content"] = full_response
                yield "", history

        if not full_response.strip():
            history[-1]["content"] = "（模型未返回有效内容）"
            yield "", history

    except Exception as e:
        error_msg = f"❌ 请求失败：{str(e)}"
        history[-1]["content"] = error_msg
        yield "", history