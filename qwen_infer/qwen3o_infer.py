import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

import torch

from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

if __name__ == '__main__':
    # vLLM engine v1 not supported yet
    os.environ['VLLM_USE_V1'] = '0'

    MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    # MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

    llm = LLM(
            model=MODEL_PATH, trust_remote_code=True, gpu_memory_utilization=0.95,
            tensor_parallel_size=torch.cuda.device_count(),
            limit_mm_per_prompt={'image': 3, 'video': 3, 'audio': 3},   # maximum number of each modality's data allowed per message
            max_num_seqs=8, # number of sequences that vLLM processes in parallel during each inference step
            max_model_len=32768,
            seed=1234,
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"}
            ], 
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

    inputs = {
        'prompt': text,
        'multi_modal_data': {},
        "mm_processor_kwargs": {
            "use_audio_in_video": True,
        },
    }

    if images is not None:
        inputs['multi_modal_data']['image'] = images
    if videos is not None:
        inputs['multi_modal_data']['video'] = videos
    if audios is not None:
        inputs['multi_modal_data']['audio'] = audios

    outputs = llm.generate([inputs], sampling_params=sampling_params)

    print(outputs[0].outputs[0].text)



# import os
# os.environ['VLLM_USE_V1'] = '0'
# os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
# import torch
# import warnings
# import numpy as np

# warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore', category=DeprecationWarning)
# warnings.filterwarnings('ignore', category=FutureWarning)
# warnings.filterwarnings('ignore', category=UserWarning)

# from qwen_omni_utils import process_mm_info
# from transformers import Qwen3OmniMoeProcessor

# def _load_model_processor():
#     if USE_TRANSFORMERS:
#         from transformers import Qwen3OmniMoeForConditionalGeneration
#         if TRANSFORMERS_USE_FLASH_ATTN2:
#             model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL_PATH,
#                                                                          dtype='auto',
#                                                                          attn_implementation='flash_attention_2',
#                                                                          device_map="auto")
#         else:
#             model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL_PATH, device_map="auto", dtype='auto')
#     else:
#         from vllm import LLM
#         model = LLM(
#             model=MODEL_PATH, trust_remote_code=True, gpu_memory_utilization=0.95,
#             tensor_parallel_size=torch.cuda.device_count(),
#             limit_mm_per_prompt={'image': 1, 'video': 3, 'audio': 3},
#             max_num_seqs=1,
#             max_model_len=32768,
#             seed=1234,
#         )

#     processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
#     return model, processor

# def run_model(model, processor, messages, return_audio, use_audio_in_video):
#     if USE_TRANSFORMERS:
#         text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
#         audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
#         inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video)
#         inputs = inputs.to(model.device).to(model.dtype)
#         text_ids, audio = model.generate(**inputs, 
#                                             thinker_return_dict_in_generate=True,
#                                             thinker_max_new_tokens=8192, 
#                                             thinker_do_sample=False,
#                                             speaker="Ethan", 
#                                             use_audio_in_video=use_audio_in_video,
#                                             return_audio=return_audio)
#         response = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#         if audio is not None:
#             audio = np.array(audio.reshape(-1).detach().cpu().numpy() * 32767).astype(np.int16)
#         return response, audio
#     else:
#         from vllm import SamplingParams
#         sampling_params = SamplingParams(temperature=1e-2, top_p=0.1, top_k=1, max_tokens=8192)
#         text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
#         inputs = {'prompt': text, 'multi_modal_data': {}, "mm_processor_kwargs": {"use_audio_in_video": use_audio_in_video}}
#         if images is not None: inputs['multi_modal_data']['image'] = images
#         if videos is not None: inputs['multi_modal_data']['video'] = videos
#         if audios is not None: inputs['multi_modal_data']['audio'] = audios
#         outputs = model.generate(inputs, sampling_params=sampling_params)
#         response = outputs[0].outputs[0].text
#         return response, None

# import librosa
# import audioread

# MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
# # MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

# USE_TRANSFORMERS = False
# TRANSFORMERS_USE_FLASH_ATTN2 = True

# model, processor = _load_model_processor()

# USE_AUDIO_IN_VIDEO = True
# RETURN_AUDIO = False

# audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/mixed_audio1.mp3"

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "audio", "audio": audio_path},
#             {"type": "text", "text": "判断说话人的国籍和性别，并告诉我音频里出现的音效是什么？"},
#         ]
#     }
# ]

# display(Audio(librosa.load(audioread.ffdec.FFmpegAudioFile(audio_path), sr=16000)[0], rate=16000))

# response, audio = run_model(model=model, messages=messages, processor=processor, return_audio=RETURN_AUDIO, use_audio_in_video=USE_AUDIO_IN_VIDEO)

# print(response)
# if audio is not None:
#     display(Audio(audio, rate=24000))
    
