import json
import os
from datetime import datetime, timedelta
from pydub import AudioSegment
from qwen_infer.qwen3o_infer_util import qwen3o_infer
# from qwen_infer.qwen3o_serve_util import qwen3o_serve_infer



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


def parse_time(clips_data):
    """
    Extract continuous time chunks from clips data.
    
    Args:
        clips_data (list): List of clip dictionaries from the JSON database
        
    Returns:
        list: List of strings representing continuous time chunks in format 
              "YYYY/MM/DD HH:MM:SS-YYYY/MM/DD HH:MM:SS"
    """
    if not clips_data:
        print("No clips data provided.")
        return []
    
    # Sort clips by start time
    sorted_clips = sorted(clips_data, key=lambda x: x['timestamp'].split('-')[0])
    
    time_chunks = []
    current_chunk_start = None
    current_chunk_end = None
    
    for clip in sorted_clips:
        start_str, end_str = clip['timestamp'].split('-')
        clip_start = datetime.strptime(start_str, "%Y/%m/%d %H:%M:%S")
        clip_end = datetime.strptime(end_str, "%Y/%m/%d %H:%M:%S")
        
        if current_chunk_start is None:
            # First clip, start new chunk
            current_chunk_start = clip_start
            current_chunk_end = clip_end
        else:
            # Check if this clip is consecutive with current chunk
            # Allow small tolerance for floating point issues (1 second)
            time_gap = (clip_start - current_chunk_end).total_seconds()
            
            if abs(time_gap) <= 1:
                # Clips are consecutive, extend current chunk
                current_chunk_end = max(current_chunk_end, clip_end)
            else:
                # Gap detected, save current chunk and start new one
                time_chunks.append(
                    f"{current_chunk_start.strftime('%Y/%m/%d %H:%M:%S')}-"
                    f"{current_chunk_end.strftime('%Y/%m/%d %H:%M:%S')}"
                )
                current_chunk_start = clip_start
                current_chunk_end = clip_end
    
    # Add the last chunk
    if current_chunk_start is not None:
        time_chunks.append(
            f"{current_chunk_start.strftime('%Y/%m/%d %H:%M:%S')}-"
            f"{current_chunk_end.strftime('%Y/%m/%d %H:%M:%S')}"
        )
    
    return time_chunks


def retrieve_time(clips_data, time_chunks):
    """
    Retrieve all clips that fall within the specified time chunks.
    
    Args:
        clips_data (list): List of clip dictionaries from the JSON database
        time_chunks (list): List of time chunk strings in format 
                           "YYYY/MM/DD HH:MM:SS-YYYY/MM/DD HH:MM:SS"
        
    Returns:
        list: List of clip timestamps that fall within the specified time chunks
    """
    if not clips_data or not time_chunks:
        print("No clips data or time chunks provided.")
        return []
    
    # Parse time chunks into datetime ranges
    chunk_ranges = []
    for chunk in time_chunks:
        start_str, end_str = chunk.split('-')
        start_time = datetime.strptime(start_str, "%Y/%m/%d %H:%M:%S")
        end_time = datetime.strptime(end_str, "%Y/%m/%d %H:%M:%S")
        chunk_ranges.append((start_time, end_time))
    
    matching_clips = []
    
    for clip in clips_data:
        clip_start_str, clip_end_str = clip['timestamp'].split('-')
        clip_start = datetime.strptime(clip_start_str, "%Y/%m/%d %H:%M:%S")
        clip_end = datetime.strptime(clip_end_str, "%Y/%m/%d %H:%M:%S")
        
        # Check if clip overlaps with any time chunk
        for chunk_start, chunk_end in chunk_ranges:
            # Clip overlaps if it starts before chunk ends and ends after chunk starts
            if clip_start <= chunk_end and clip_end >= chunk_start:
                matching_clips.append(clip['timestamp'])
                break  # No need to check other chunks for this clip
    
    return matching_clips


def label_clips_1(clips_data, json_path, llm_infer_function):
    """
    Generate label1 for audio clips that don't have labels using LLM.
    
    Args:
        clips_data (list): List of clip dictionaries from the JSON database
        json_path (str): Path to JSON database file
        llm_infer_function (function): LLM inference function that takes messages_list
        
    Returns:
        list: Updated clips_data with generated label1 for previously empty entries
    """
    
    # Filter clips that need labeling
    clips_to_label = [clip for clip in clips_data if not clip.get('label1', '').strip()]
    
    if not clips_to_label:
        print("No clips need labeling - all label1 fields are already filled")
        return clips_data
    
    print(f"Generating level 1 labels for {len(clips_to_label)} clips...")
    
    # Prepare conversations for LLM
    messages_list = []
    
    for clip in clips_to_label:
        # Get audio file path
        audio_path = clip['path']
        
        # Check if file exists
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            continue
            
        # Create conversation for this clip
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are analyzing audio recordings from a wearable device. Provide a concise, coarse-grained description of the main scenario, place, and activity in the audio. Use only a few words, not exceeding one sentence. Examples: 'office, working', 'subway station, commuting', 'outdoor by the road, chat', 'restaurant, eating', 'park, walking'."}
                ]
            },
            {
                "role": "user", 
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": "Describe the main scenario or place, and what is happening in this audio clip in a few concise words."}
                ]
            }
        ]
        
        messages_list.append(conversation)
    
    # Call LLM for inference
    try:
        results = llm_infer_function(messages_list)
        
        # Update clips_data with generated labels
        result_index = 0
        for clip in clips_data:
            if not clip.get('label1', '').strip():
                if result_index < len(results):
                    # Extract the label from LLM response
                    llm_response = results[result_index]
                    # Clean up the response to ensure it's concise
                    label = clean_label_response(llm_response)
                    clip['label1'] = label
                    print(f"Labeled clip {result_index + 1}: {label}")
                    result_index += 1
                    
    except Exception as e:
        print(f"Error during LLM inference: {e}")
        
    with open(json_path, 'w') as jf:
        json.dump(clips_data, jf, indent=2)
    
    return clips_data


def clean_label_response(llm_response):
    """
    Clean and format the LLM response to ensure it's concise.
    
    Args:
        llm_response (str): Raw response from LLM
        
    Returns:
        str: Cleaned, concise label
    """
    # Remove any quotation marks
    cleaned = llm_response.strip('"\'').strip()
    
    # If response is too long, take only the first sentence
    if len(cleaned) > 100:
        # Split by common sentence endings and take first part
        for delimiter in ['.', ';', '-']:
            if delimiter in cleaned:
                cleaned = cleaned.split(delimiter)[0].strip()
                break
        # If still too long, truncate
        if len(cleaned) > 100:
            cleaned = cleaned[:97] + "..."
    
    return cleaned



database_json_path = "/aiot-nvme-15T-x2-hk01/home/aiot25-suyihang/AudioDaily/test_results/metadata.json"
metadata = chunk_audio_clips("/aiot-nvme-15T-x2-hk01/home/aiot25-suyihang/AudioDaily/test_audios/251111_201000.WAV",
                             database_json_path,
                             "/aiot-nvme-15T-x2-hk01/home/aiot25-suyihang/AudioDaily/test_results")
labeled_metadata_1 = label_clips_1(metadata, database_json_path, qwen3o_infer)