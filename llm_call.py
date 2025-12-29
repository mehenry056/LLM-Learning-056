import os
from typing import List, Dict, Callable


# Function 1: Call any popular LLM
def call_llm(provider: str, model: str, prompt: str, api_key: str, **kwargs) -> str:
    """
    Calls a popular LLM based on the provider.

    Supported providers: 'openai', 'anthropic', 'google' (for Gemini), 'groq' (for various models).

    Args:
        provider (str): The LLM provider (e.g., 'openai', 'anthropic').
        model (str): The model name (e.g., 'gpt-4o', 'claude-3-sonnet-20240229').
        prompt (str): The input prompt.
        api_key (str): API key for the provider.
        **kwargs: Additional parameters like temperature, max_tokens, etc.

    Returns:
        str: The generated response.

    Example:
        response = call_llm('openai', 'gpt-4o', 'Hello!', 'your_openai_key', temperature=0.7)
    """
    if provider.lower() == 'openai':
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

    elif provider.lower() == 'anthropic':
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=kwargs.get('max_tokens', 1024),
            messages=[{"role": "user", "content": prompt}],
            **{k: v for k, v in kwargs.items() if k != 'max_tokens'}
        )
        return response.content[0].text

    elif provider.lower() == 'google':
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Install google-generative-ai: pip install google-generative-ai")

        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(prompt, generation_config=kwargs)
        return response.text

    elif provider.lower() == 'groq':
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Install groq: pip install groq")

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content
    elif provider.lower() == 'dashscope-sdk':
        try:
            import dashscope
            from dashscope import Generation
        except ImportError:
            raise ImportError("请安装: pip install dashscope")

        dashscope.api_key = api_key
        response = Generation.call(
            model=model,  # 如 "qwen-max"
            messages=[{'role': 'user', 'content': prompt}],
            result_format='message',
            **kwargs  # temperature 等参数也支持
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            raise Exception(f"通义千问调用失败: {response.message}")
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported: openai, anthropic, google, groq.")