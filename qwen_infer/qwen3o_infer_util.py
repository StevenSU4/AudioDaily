import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'

import torch

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
import re

def build_input(processor, messages, use_audio_in_video):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)

    inputs = {
        'prompt': text,
        'multi_modal_data': {},
        "mm_processor_kwargs": {
            "use_audio_in_video": use_audio_in_video,
        },
    }

    if images is not None:
        inputs['multi_modal_data']['image'] = images
    if videos is not None:
        inputs['multi_modal_data']['video'] = videos
    if audios is not None:
        inputs['multi_modal_data']['audio'] = audios
    
    return inputs

def qwen3o_infer(messages: list, instruct_model=False, use_audio_in_video=True):
    # vLLM engine v1 not supported yet
    os.environ['VLLM_USE_V1'] = '0'

    if instruct_model:
        MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    else:
        MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

    llm = LLM(
            model=MODEL_PATH, trust_remote_code=True, gpu_memory_utilization=0.95,
            tensor_parallel_size=torch.cuda.device_count(),
            limit_mm_per_prompt={'image': 0, 'video': 0, 'audio': 3},
            max_num_seqs=8,
            max_model_len=65536, # 32768
            seed=1234,
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    # Combine messages for batch processing
    inputs = [build_input(processor, message, use_audio_in_video) for message in messages]

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    result = [outputs[i].outputs[0].text for i in range(len(outputs))]
    
    if instruct_model:
        return result
    else:
        _pattern = re.compile(r'(?is)^\s*<think>.*?(?:</think>|<\\think>)\s*')
        result = [_pattern.sub('', r).lstrip() for r in result]
        return result
    
    
# use case:

# # Conversation with audio only
# conversation2 = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "audio", "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"},
#             {"type": "text", "text": "What can you hear in this audio?"},
#         ]
#     }
# ]

# # Conversation with pure text and system prompt
# conversation3 = [
#     {
#         "role": "system",
#         "content": [
#             {"type": "text", "text": "You are Qwen-Omni."}
#         ],
#     },
#     {
#         "role": "user",
#         "content": "Who are you? Answer in one sentence."
#     }
# ]

# # Conversation with mixed media
# conversation4 = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"},
#             {"type": "audio", "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_fr.wav"},
#             {"type": "text", "text": "What can you see and hear? Answer in one sentence."}
#         ],
#     }
# ]

# messages_list = [conversation2, conversation3]

# result = qwen3o_infer(messages_list)

# print("Results:")
# for i, res in enumerate(result):
#     print(f"Conversation {i+1} Response: {res}")