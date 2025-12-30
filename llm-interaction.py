import gradio as gr
import statics
import llm_call


def update_model_list(provider_choice):
    provider_key = provider_choice.lower()
    models = statics.MODEL_MAP.get(provider_key, statics.MODEL_MAP["dashscope"])
    return gr.Dropdown(choices=models, value=models[0], label="模型")


def clear_chat():
    return None, []

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
                choices=list(statics.MODEL_MAP.keys()),
                value="dashscope",
                label="模型提供商",
                info="选择 API 服务商"
            )
            model = gr.Dropdown(
                choices=statics.MODEL_MAP["dashscope"],
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
        fn=llm_call.chat_with_llm,
        inputs=[msg, chatbot, provider, model,  ],
        outputs=[msg, chatbot]
    )
    submit_btn.click(
        fn=llm_call.chat_with_llm,
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