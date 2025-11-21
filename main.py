# from qwen_infer.qwen3o_infer_util import qwen3o_infer

# # Conversation with image only
# conversation1 = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"},
#             {"type": "text", "text": "What can you see in this image? Answer in one sentence."},
#         ]
#     }
# ]

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

import json
import os
from pydub import AudioSegment
from pydub.utils import make_chunks

def process_audio_clips(input_audio_path, output_dir, clip_length=30, overlap=5):
    """
    Process an audio file into overlapping clips and create a JSON database.
    
    Args:
        input_audio_path (str): Path to input audio file
        output_dir (str): Directory to save audio clips and JSON database
        clip_length (int): Length of each clip in seconds (default: 30)
        overlap (int): Overlap between clips in seconds (default: 5)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio file
    audio = AudioSegment.from_file(input_audio_path)
    duration_sec = len(audio) / 1000  # pydub works in milliseconds
    
    # Calculate step size (non-overlapping portion)
    step = clip_length - overlap
    clips_data = []
    
    # Generate clips
    start = 0
    clip_index = 0
    
    while start < duration_sec:
        end = min(start + clip_length, duration_sec)
        
        # Extract clip
        clip_start_ms = start * 1000
        clip_end_ms = end * 1000
        clip = audio[clip_start_ms:clip_end_ms]
        
        # Create filename with timestamps
        start_str = f"{int(start//60):02d}_{int(start%60):02d}"
        end_str = f"{int(end//60):02d}_{int(end%60):02d}"
        filename = f"clip_{start_str}_{end_str}.wav"
        filepath = os.path.join(output_dir, filename)
        
        # Export clip
        clip.export(filepath, format="wav")
        
        # Add to database
        clip_data = {
            "path": filepath,
            "start_time": start,
            "end_time": end,
            "label1": "",
            "label2": ""
        }
        clips_data.append(clip_data)
        
        # Move to next segment
        start += step
        clip_index += 1
    
    # Save JSON database
    json_path = os.path.join(output_dir, "audio_clips_database.json")
    with open(json_path, 'w') as f:
        json.dump(clips_data, f, indent=2)
    
    return clips_data