# 每个提供商对应的 API Base URL（OpenAI 兼容模式）
BASE_URL_MAP = {
    "openai":    "https://api.openai.com/v1",                  # 官方 OpenAI
    "groq":      "https://api.groq.com/openai/v1",             # Groq 官方
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 通义千问兼容模式
    "gemini":    "https://generativelanguage.googleapis.com/v1beta",   # Google Gemini（注意：不是完全 OpenAI 兼容，需要特殊处理或用官方 SDK）
    "grok":      "https://api.x.ai/v1"                        # xAI Grok 官方（完全兼容 OpenAI SDK）
}

# 对应的环境变量 API Key 名称
API_KEY_ENV = {
    "openai":    "OPENAI_API_KEY",
    "groq":      "GROQ_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "gemini":    "GEMINI_API_KEY",      # 或 GOOGLE_API_KEY，Google 官方推荐
    "grok":      "XAI_API_KEY"          # xAI 官方文档指定的环境变量名
}

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