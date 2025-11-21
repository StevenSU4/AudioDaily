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
from datetime import datetime, timedelta
from pydub import AudioSegment

def chunk_audio_clips(input_audio_path, json_path, output_dir, clip_length=30, overlap=5):
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
    
    # Extract timestamp from input filename (without extension)
    input_filename = os.path.basename(input_audio_path)
    filename_without_ext = os.path.splitext(input_filename)[0]
    
    # Parse the timestamp from filename (format: "251110_203015")
    try:
        # Convert to datetime object
        base_datetime = datetime.strptime(filename_without_ext, "%y%m%d_%H%M%S")
    except ValueError:
        raise ValueError(f"Filename '{filename_without_ext}' doesn't match expected format 'yymmdd_HHMMSS'")
    
    # Load audio file
    audio = AudioSegment.from_file(input_audio_path)
    duration_sec = len(audio) / 1000  # pydub works in milliseconds
    
    # Calculate step size (non-overlapping portion)
    step = clip_length - overlap
    
    # Load existing clips_data if json exists
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as jf:
                clips_data = json.load(jf)
                if not isinstance(clips_data, list):
                    print("Invalid JSON format, starting with empty clips_data")
                    clips_data = []
        except (json.JSONDecodeError, OSError):
            print("Error reading JSON file, starting with empty clips_data")
            clips_data = []
    else:
        clips_data = []
    
    # Generate clips
    start_offset = 0
    clip_index = 0
    
    while start_offset < duration_sec:
        end_offset = min(start_offset + clip_length, duration_sec)
        
        # Calculate actual start and end timestamps
        clip_start_time = base_datetime + timedelta(seconds=start_offset)
        clip_end_time = base_datetime + timedelta(seconds=end_offset)
        
        # Create filename using start timestamp only
        clip_filename = clip_start_time.strftime("%y%m%d_%H%M%S") + ".wav"
        filepath = os.path.join(output_dir, clip_filename)
        
        # Extract and export clip
        clip_start_ms = start_offset * 1000
        clip_end_ms = end_offset * 1000
        clip = audio[clip_start_ms:clip_end_ms]
        clip.export(filepath, format="wav")
        
        # Create timestamp string for JSON (format: "2025/11/10 20:30:15-2025/11/10 20:30:45")
        timestamp_str = (f"{clip_start_time.strftime('%Y/%m/%d %H:%M:%S')}-"
                        f"{clip_end_time.strftime('%Y/%m/%d %H:%M:%S')}")
        
        # Add to database
        clip_data = {
            "path": filepath,
            "timestamp": timestamp_str,
            "label1": "",
            "label2": ""
        }
        clips_data.append(clip_data)
        
        # Move to next segment
        start_offset += step
        clip_index += 1
    
    # Save JSON database
    with open(json_path, 'w') as jf:
        json.dump(clips_data, jf, indent=2)
    
    print(f"Processed {len(clips_data)} clips")
    return clips_data

