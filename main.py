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


def label_clips_2(clips_data, target_timestamps, llm_infer_function):
    """
    Generate detailed label2 for specific clips and validate/update label1 if needed.
    
    Args:
        clips_data (list): List of clip dictionaries from the JSON database
        target_timestamps (list): List of timestamps to process
        llm_infer_function (function): LLM inference function
        
    Returns:
        list: Updated clips_data with generated label2 and validated/updated label1
    """
    
    # Filter clips that match target timestamps and don't have label2
    clips_to_label = []
    for clip in clips_data:
        if (clip['timestamp'] in target_timestamps and 
            not clip.get('label2', '').strip()):
            clips_to_label.append(clip)
    
    if not clips_to_label:
        print("No target clips need label2 generation")
        return clips_data
    
    print(f"Generating detailed labels for {len(clips_to_label)} target clips...")
    
    # Prepare conversations for LLM
    messages_list = []
    
    for clip in clips_to_label:
        audio_path = clip['path']
        current_label1 = clip.get('label1', '')
        
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            continue
            
        # Create detailed system prompt
        system_prompt = f"""
You are analyzing audio recordings from a wearable device. For the given audio clip:

1. Generate a DETAILED description (label2) that includes:
   - The specific scenario or environment
   - The place/location with as much detail as possible
   - Events that are happening
   - Any speech content (transcribe if possible)
   - What the user is likely doing
   This should be fine-grained enough to reconstruct the user's daily life.

2. Check the existing coarse label (label1): "{current_label1}"
   - If this label accurately represents the main scenario/activity, keep it as is
   - If it's incorrect or incomplete, update it to a concise version (few words, one sentence max)
   - The updated label1 should follow the format: "location, activity" (e.g., "coffee shop, ordering drink")

Provide your response in this exact format:
LABEL2: [your detailed description here]
LABEL1: [existing label1 if correct, or updated concise version]
"""
        
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt.strip()}
                ]
            },
            {
                "role": "user", 
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": "Analyze this audio clip and provide both detailed and concise labels as instructed."}
                ]
            }
        ]
        
        messages_list.append(conversation)
    
    # Call LLM for inference
    try:
        results = llm_infer_function(messages_list)
        
        # Process results and update clips_data
        result_index = 0
        for clip in clips_data:
            if (clip['timestamp'] in target_timestamps and 
                not clip.get('label2', '').strip() and 
                result_index < len(results)):
                
                llm_response = results[result_index]
                label2, updated_label1 = parse_label_response(llm_response, clip.get('label1', ''))
                
                # Update the clip
                clip['label2'] = label2
                if updated_label1 and updated_label1 != clip.get('label1', ''):
                    print(f"Updated label1 for clip {result_index + 1}: '{clip['label1']}' -> '{updated_label1}'")
                    clip['label1'] = updated_label1
                
                print(f"Processed clip {result_index + 1}:")
                print(f"  Label1: {clip['label1']}")
                print(f"  Label2: {label2[:100]}..." if len(label2) > 100 else f"  Label2: {label2}")
                print()
                
                result_index += 1
                
    except Exception as e:
        print(f"Error during LLM inference: {e}")
        
    with open(json_path, 'w') as jf:
        json.dump(clips_data, jf, indent=2)
        
    return clips_data


def parse_label_response(llm_response, current_label1):
    """
    Parse the LLM response to extract label2 and updated label1.
    
    Args:
        llm_response (str): Raw response from LLM
        current_label1 (str): Current label1 value
        
    Returns:
        tuple: (label2, label1) where label1 is updated if needed, otherwise current_label1
    """
    lines = llm_response.strip().split('\n')
    label2 = ""
    label1 = current_label1
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('LABEL2:') or line.startswith('Label2:') or line.startswith('label2:'):
            # Extract label2 (might span multiple lines)
            label2_parts = [line.split(':', 1)[1].strip()]
            i += 1
            # Continue reading subsequent lines until we hit LABEL1 or end
            while i < len(lines) and not (
                lines[i].strip().startswith('LABEL1:') or 
                lines[i].strip().startswith('Label1:') or 
                lines[i].strip().startswith('label1:')
            ):
                label2_parts.append(lines[i].strip())
                i += 1
            label2 = ' '.join(label2_parts).strip()
            
        elif line.startswith('LABEL1:') or line.startswith('Label1:') or line.startswith('label1:'):
            new_label1 = line.split(':', 1)[1].strip()
            # Only update if different from current and not empty
            if new_label1 and new_label1 != current_label1:
                label1 = clean_label_response(new_label1)
            i += 1
        else:
            i += 1
    
    # If we didn't find structured response, try to parse differently
    if not label2:
        # Look for the most detailed part as label2
        paragraphs = [p.strip() for p in llm_response.split('\n\n') if p.strip()]
        if len(paragraphs) >= 2:
            label2 = paragraphs[0]  # Assume first paragraph is detailed description
            # Last part might be the concise label
            last_part = paragraphs[-1]
            if len(last_part) < 100:  # Likely the concise label
                label1 = clean_label_response(last_part)
        elif paragraphs:
            label2 = llm_response
    
    return label2, label1


def main_qa_system(clips_data, llm_infer_function, user_question=None):
    """
    Main QA system that answers user questions about daily life using audio recordings.
    
    Args:
        clips_data (list): Audio clips database
        llm_infer_function (function): LLM inference function
        user_question (str): User's question (if None, will prompt for input)
        
    Returns:
        str: Final answer to user's question
    """
    
    # Get user question if not provided
    if user_question is None:
        user_question = input("Please enter your question about your daily life: ")
    
    print(f"User question: {user_question}")
    print("Processing...")
    
    # Step 1: Get time chunks covered in database
    available_time_chunks = parse_time(clips_data)
    print(f"Available time chunks in database: {len(available_time_chunks)} chunks")
    
    # Step 2: Ask LLM which time chunks are needed to answer the question
    needed_time_chunks = ask_llm_for_relevant_time_chunks(
        available_time_chunks, user_question, llm_infer_function
    )
    
    if not needed_time_chunks:
        return "I couldn't find relevant time periods in the available data to answer your question."
    
    print(f"LLM identified {len(needed_time_chunks)} relevant time chunks")
    
    # Step 3: Retrieve clips within the needed time chunks
    relevant_clips = retrieve_clips_by_time_chunks(clips_data, needed_time_chunks)
    print(f"Retrieved {len(relevant_clips)} relevant audio clips")
    
    # Step 4: Ask LLM if it can answer with label1 information
    initial_response = ask_llm_initial_analysis(relevant_clips, user_question, llm_infer_function)
    
    # Step 5: Check LLM response
    if initial_response.startswith("ANSWER_FOUND"):
        # Extract and return the answer
        answer = extract_answer_from_response(initial_response)
        return answer
    
    elif initial_response.startswith("ANSWER_NOT_FOUND"):
        # Extract timestamps that need more detail
        needed_timestamps = extract_timestamps_from_response(initial_response)
        print(f"LLM needs more detailed information for {len(needed_timestamps)} clips")
        
        # Step 6: Generate label2 for the needed clips
        updated_clips_data = label_clips_2(clips_data, needed_timestamps, llm_infer_function)
        
        # Step 7: Get final answer with detailed label2 information
        final_answer = ask_llm_final_analysis(
            updated_clips_data, needed_timestamps, user_question, llm_infer_function
        )
        return final_answer
    
    else:
        return "I encountered an issue while processing your question. Please try again."


def ask_llm_for_relevant_time_chunks(available_time_chunks, user_question, llm_infer_function):
    """
    Ask LLM to identify which time chunks are relevant to answer the user's question.
    
    Args:
        available_time_chunks (list): List of time chunks from parse_time
        user_question (str): User's question
        llm_infer_function (function): LLM inference function
        
    Returns:
        list: Relevant time chunks in format "YYYY/MM/DD HH:MM:SS-YYYY/MM/DD HH:MM:SS"
    """
    
    system_prompt = """
You are an AI assistant that helps identify relevant time periods from audio recordings to answer questions about daily life.

Given a user's question and a list of available time chunks from audio recordings, you need to return a list of time chunks that are most likely to contain information needed to answer the question.

Return ONLY a Python list of strings in the format: ["YYYY/MM/DD HH:MM:SS-YYYY/MM/DD HH:MM:SS", ...]

Rules:
1. Only return time chunks that exist in the available time chunks
2. Be specific about time ranges relevant to the question
3. Consider the context of the question (e.g., meal times, work hours, sleep, etc.)
4. If no relevant time chunks exist, return an empty list
5. Your response should contain ONLY the Python list, no other text
"""

    user_content = f"""
User's question: {user_question}

Available time chunks in database:
{available_time_chunks}

Return the relevant time chunks as a Python list:
"""
    
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user", 
            "content": [{"type": "text", "text": user_content}]
        }
    ]
    
    try:
        response = llm_infer_function([conversation])[0]
        # Parse the response as a Python list
        import ast
        time_chunks = ast.literal_eval(response)
        return time_chunks
    except Exception as e:
        print(f"Error parsing LLM response for time chunks: {e}")
        return []


def retrieve_clips_by_time_chunks(clips_data, time_chunks):
    """
    Retrieve full clip data for clips that fall within the given time chunks.
    
    Args:
        clips_data (list): List of clip dictionaries
        time_chunks (list): List of time chunk strings
        
    Returns:
        list: Clip dictionaries that fall within the time chunks
    """
    return retrieve_time_full(clips_data, time_chunks)


def retrieve_time_full(clips_data, time_chunks):
    """
    Extended version of retrieve_time that returns full clip data instead of just timestamps.
    """
    if not clips_data or not time_chunks:
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
            if clip_start <= chunk_end and clip_end >= chunk_start:
                matching_clips.append(clip)
                break
    
    return matching_clips


def ask_llm_initial_analysis(clips, user_question, llm_infer_function):
    """
    Ask LLM if it can answer the question with the provided clips' label1 information.
    
    Args:
        clips (list): Relevant clip dictionaries
        user_question (str): User's question
        llm_infer_function (function): LLM inference function
        
    Returns:
        str: LLM response starting with either "ANSWER_FOUND" or "ANSWER_NOT_FOUND"
    """
    
    system_prompt = """
You are analyzing audio recordings from a wearable device to answer questions about the user's daily life.

You will be provided with:
1. The user's question
2. A list of audio clips with their timestamps and coarse labels (label1)

Your task:
1. First, determine if you can answer the user's question based on the coarse labels (label1) alone.
2. If YES, respond with: "ANSWER_FOUND: [your answer to the question]"
3. If NO, respond with: "ANSWER_NOT_FOUND: [list of timestamps that need more detailed analysis]"

Rules for ANSWER_FOUND:
- Provide a clear, concise answer to the user's question
- Base your answer only on the information in the coarse labels
- Start exactly with "ANSWER_FOUND:"

Rules for ANSWER_NOT_FOUND:
- List the specific timestamps (in the same format as provided) that need detailed label2 information
- Only include timestamps that are crucial for answering the question
- Start exactly with "ANSWER_NOT_FOUND:"
- Format: "ANSWER_NOT_FOUND: ['timestamp1', 'timestamp2', ...]"
"""

    # Prepare clips information
    clips_info = []
    for clip in clips:
        clips_info.append(f"Timestamp: {clip['timestamp']}, Label1: {clip.get('label1', 'N/A')}")
    
    clips_text = "\n".join(clips_info)
    
    user_content = f"""
User's question: {user_question}

Available audio clips with coarse labels:
{clips_text}

Can you answer the question based on this information?
"""
    
    conversation = [
        {
            "role": "system", 
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_content}]
        }
    ]
    
    response = llm_infer_function([conversation])[0]
    return response


def extract_answer_from_response(response):
    """Extract the answer from ANSWER_FOUND response."""
    if "ANSWER_FOUND:" in response:
        return response.split("ANSWER_FOUND:", 1)[1].strip()
    return response


def extract_timestamps_from_response(response):
    """Extract timestamps from ANSWER_NOT_FOUND response."""
    try:
        if "ANSWER_NOT_FOUND:" in response:
            import ast
            timestamp_str = response.split("ANSWER_NOT_FOUND:", 1)[1].strip()
            return ast.literal_eval(timestamp_str)
    except Exception as e:
        print(f"Error extracting timestamps: {e}")
    return []


def ask_llm_final_analysis(clips_data, target_timestamps, user_question, llm_infer_function):
    """
    Ask LLM for final answer using detailed label2 information.
    
    Args:
        clips_data (list): Updated clips data with label2
        target_timestamps (list): Timestamps that have detailed labels
        user_question (str): User's question
        llm_infer_function (function): LLM inference function
        
    Returns:
        str: Final answer to user's question
    """
    
    # Get the clips with detailed information
    detailed_clips = []
    for clip in clips_data:
        if clip['timestamp'] in target_timestamps and clip.get('label2'):
            detailed_clips.append(clip)
    
    system_prompt = """
You are analyzing detailed audio recordings from a wearable device to answer questions about the user's daily life.

You have been provided with detailed descriptions (label2) of specific audio clips that contain the information needed to answer the user's question.

Please provide a comprehensive answer to the user's question based on the detailed information available.
"""

    # Prepare detailed clips information
    clips_info = []
    for clip in detailed_clips:
        clips_info.append(f"""
Timestamp: {clip['timestamp']}
Coarse Label: {clip.get('label1', 'N/A')}
Detailed Description: {clip.get('label2', 'N/A')}
""")
    
    clips_text = "\n---\n".join(clips_info)
    
    user_content = f"""
User's question: {user_question}

Detailed information from relevant audio clips:
{clips_text}

Please provide a comprehensive answer to the user's question based on this detailed information.
"""
    
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_content}]
        }
    ]
    
    response = llm_infer_function([conversation])[0]
    return response


# Example usage and main program entry point
if __name__ == "__main__":
    # Load your clips data
    try:
        with open('audio_clips_database.json', 'r') as f:
            clips_data = json.load(f)
    except FileNotFoundError:
        print("Error: audio_clips_database.json not found. Please process your audio files first.")
        exit(1)
    
    # Initialize your LLM inference function (replace with your actual function)
    # qwen3o_infer = your_llm_inference_function
    
    # Run the QA system
    print("=== Daily Life Audio QA System ===")
    print("This system answers questions about your daily life using audio recordings from your wearable device.")
    print()
    
    while True:
        user_question = input("\nEnter your question (or 'quit' to exit): ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            break
            
        if not user_question:
            continue
            
        # In a real implementation, you would call:
        # answer = main_qa_system(clips_data, qwen3o_infer, user_question)
        # print(f"\nAnswer: {answer}")
        
        # For demo purposes:
        print("\n[This would call the main_qa_system with your LLM function]")
        print(f"Question: {user_question}")
        print("System would process this using the available audio clips and provide an answer.")



database_json_path = "/aiot-nvme-15T-x2-hk01/home/aiot25-suyihang/AudioDaily/test_results/metadata.json"
metadata = chunk_audio_clips("/aiot-nvme-15T-x2-hk01/home/aiot25-suyihang/AudioDaily/test_audios/251111_210000.WAV",
                             database_json_path,
                             "/aiot-nvme-15T-x2-hk01/home/aiot25-suyihang/AudioDaily/test_results")
labeled_metadata_1 = label_clips_1(metadata, database_json_path, qwen3o_infer)


