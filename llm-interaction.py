import gradio as gr
import os
from openai import OpenAI

# 常量定义
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENAI_BASE_URL = "https://platform.openai.com/docs"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"   # Google Gemini（注意：不是完全 OpenAI 兼容，需要特殊处理或用官方 SDK）
GROK_BASE_URL = "https://api.x.ai/v1"                        # xAI Grok 官方（完全兼容 OpenAI SDK）

# 模型映射表
MODEL_MAP = {
    "openai": [
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
        "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
    ],
    "groq": [
        "llama-3.1-70b-versatile", "llama-3.1-8b-instant",
        "llama3-70b-8192", "llama3-8b-8192",
        "mixtral-8x7b-32768", "gemma-7b-it"
    ],
    "dashscope": [
        "qwen-max", "qwen-plus", "qwen-turbo",
        "qwen-long", "qwen-vl-max", "qwen-vl-plus"
    ],
    "gemini": [                               # Google Gemini 系列
        "gemini-1.5-pro",                     # 目前最强，推荐主力使用
        "gemini-1.5-flash",                   # 速度快、性价比高
        "gemini-1.0-pro",                     # 旧版稳定版（部分场景仍可用）
        "gemini-pro-vision"                   # 多模态（支持图片）
    ],
    "grok": [                                 # xAI Grok 系列
        "grok-4",                             # 最新最强模型（逐步开放中）
        "grok-3",                             # 高性能主力模型
        "grok-3-mini",                        # 轻量快速版
        "grok-2",                             # 早期版本（兼容性好）
        "grok-2-vision"                       # 支持图像的多模态版本
    ]
}


def update_model_list(provider_choice):
    provider_key = provider_choice.lower()
    models = MODEL_MAP.get(provider_key, MODEL_MAP["dashscope"])
    return gr.Dropdown(choices=models, value=models[0], label="模型")


def clear_chat():
    return None, []


def chat_with_llm(message, history, provider, model, temperature):
    if not message or not message.strip():
        yield "", history
        return

    # 1. 这里的 history 现在是一个 list of dicts: [{'role': 'user', 'content': '...'}, ...]
    provider = provider.lower()
    api_key_env = {
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "dashscope": "DASHSCOPE_API_KEY",
        "gemini": "GEMINI_API_KEY",  # 或 GOOGLE_API_KEY（Google 官方推荐）
        "grok": "XAI_API_KEY"  # xAI 官方使用的环境变量名
    }.get(provider)

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
            client = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)
            extra_body = {"enable_search": True}
        elif provider == "groq":
            client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
            extra_body = {}
        elif provider == "gemini":
            client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
            extra_body = {}
        elif provider == "grok":
            client = OpenAI(api_key=api_key, base_url=GROK_BASE_URL)
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


# Gradio 6.0+ 兼容写法
with gr.Blocks() as demo:  # 移除了 theme 和 title 参数
    gr.Markdown("""
    # 🤖 🍎多模型 AI 聊天助手

    支持 OpenAI、Groq、通义千问（DashScope）等多种模型一键切换  
    请提前在环境变量中配置对应的 API Key
    """)

    with gr.Row():
        with gr.Column(scale=4):
            # 移除 type="tuples" 参数（新版默认就是 tuples）
            chatbot = gr.Chatbot(
                height=600,
                show_label=False,
                avatar_images=(None, "https://avatars.githubusercontent.com/u/148468537?s=200&v=4")
            )
            msg = gr.Textbox(
                label="输入你的问题",
                placeholder="在这里输入消息，然后按回车或点击发送...",
                lines=3
            )

            with gr.Row():
                submit_btn = gr.Button("🚀 发送", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")

        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### ⚙️ 配置")
            provider = gr.Dropdown(
                choices=list(MODEL_MAP.keys()),
                value="dashscope",
                label="模型提供商",
                info="选择 API 服务商"
            )
            model = gr.Dropdown(
                choices=MODEL_MAP["dashscope"],
                value="qwen-max",
                label="模型"
            )
            temperature = gr.Slider(
                minimum=0,
                maximum=1.5,
                value=0.7,
                step=0.1,
                label="温度 (Temperature)",
                info="值越高越有创造性，越低越确定性"
            )

            gr.Markdown("### ℹ️ 使用提示")
            gr.Markdown("""
            - OpenAI → `OPENAI_API_KEY`
            - Groq → `GROQ_API_KEY`  
            - 通义千问 → `DASHSCOPE_API_KEY`
            """)

    # 事件绑定
    provider.change(
        fn=update_model_list,
        inputs=provider,
        outputs=model
    )

    msg.submit(
        fn=chat_with_llm,
        inputs=[msg, chatbot, provider, model, temperature],
        outputs=[msg, chatbot]
    )
    submit_btn.click(
        fn=chat_with_llm,
        inputs=[msg, chatbot, provider, model, temperature],
        outputs=[msg, chatbot]
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, msg]
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,          # 如果需要公网访问，可以改为 True
        inbrowser=True,       # 自动打开浏览器
        theme=gr.themes.Soft()  # 美化主题（推荐保留）
    )