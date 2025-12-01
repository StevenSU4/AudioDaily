import requests
import json
import re

# Use this command to start vllm serving:
# CUDA_VISIBLE_DEVICES=2,3 vllm serve Qwen/Qwen3-Omni-30B-A3B-Thinking --device cuda --port 8901 --host 127.0.0.1 --dtype bfloat16 --max-model-len 32768 --allowed-local-media-path / -tp 2
# CUDA_VISIBLE_DEVICES=1,2,3,7 vllm serve Qwen/Qwen3-Omni-30B-A3B-Thinking --device cuda --port 8901 --host 127.0.0.1 --dtype bfloat16 --max-model-len 65536 --allowed-local-media-path / -tp 4
def qwen3o_serve_infer(
    system_message="You are a helpful assistant.",
    user_content=None,
    api_url="http://localhost:8901/v1/chat/completions"
) -> str:
    """
    Query a locally hosted LLM with multimodal capabilities.
    
    Args:
        system_message (str): The system prompt/role definition
        user_content (list): List of message components (text, image_url, audio_url)
        api_url (str): URL of the local LLM API endpoint
        
    Returns:
        str: The response from the LLM
    """
    
    # Default user content if none provided
    if user_content is None:
        user_content = [
            {"type": "text", "text": "Hello! How are you?"}
        ]
    
    # Construct the request payload
    payload = {
        "messages": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
    }
    
    try:
        # Send POST request to the local LLM API
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60  # 60 second timeout
        )
        
        # Raise an exception for bad status codes
        response.raise_for_status()
        
        response_text = response.json()["choices"][0]["message"]["content"]
        
        # Remove <think> part from the response
        _pattern = re.compile(r'(?is)^\s*<think>.*?(?:</think>|<\\think>)\s*')
        response_text = _pattern.sub('', response_text).lstrip()
        return response_text
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling LLM API: {e}")
        return None


# use case:

# multimodal_content = [
#     {
#         "type": "image_url",
#         "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"}
#     },
#     {
#         "type": "audio_url", 
#         "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"}
#     },
#     {
#         "type": "text",
#         "text": "What can you see and hear? Answer in one sentence."
#     }
# ]

# result = qwen3o_serve_infer(system_message="You are a helpful assistant.", user_content=multimodal_content)

# print("LLM response:", result)