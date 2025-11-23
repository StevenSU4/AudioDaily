import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

import torch

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