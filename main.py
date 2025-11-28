import json
import sys
from multiprocessing import get_context, Queue
import os
from datetime import datetime, timedelta
from pydub import AudioSegment
from qwen_infer.qwen3o_infer_util import qwen3o_infer
# from qwen_infer.qwen3o_serve_util import qwen3o_serve_infer

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m' 

AudioDaily_normal = f"{Colors.BOLD}{Colors.GREEN}[AudioDaily]{Colors.END}"
AudioDaily_error = f"{Colors.BOLD}{Colors.RED}[AudioDaily ERROR]{Colors.END}"



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
        print(f"{AudioDaily_error} No clips data provided.")
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
            
            if abs(time_gap) <= 10:
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


def label_clips_2(clips_data, target_timestamps, llm_infer_function, json_path):
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
        print(f"{AudioDaily_error} No target clips need label2 generation")
        return clips_data
    
    print(f"{AudioDaily_normal} Generating detailed labels for {len(clips_to_label)} target clips...")
    
    # Prepare conversations for LLM
    messages_list = []
    
    for clip in clips_to_label:
        audio_path = clip['path']
        current_label1 = clip.get('label1', '')
        
        if not os.path.exists(audio_path):
            print(f"{AudioDaily_error} Audio file not found: {audio_path}")
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
                    print(f"{AudioDaily_normal} Updated label1 for clip {result_index + 1}: '{clip['label1']}' -> '{updated_label1}'")
                    clip['label1'] = updated_label1
                
                print(f"{AudioDaily_normal} Processed clip {result_index + 1}:")
                print(f"{AudioDaily_normal}  Label1: {clip['label1']}")
                print(f"{AudioDaily_normal}  Label2: {label2[:100]}..." if len(label2) > 100 else f"{AudioDaily_normal}  Label2: {label2}")
                
                result_index += 1
                
    except Exception as e:
        print(f"{AudioDaily_error} Error during LLM inference: {e}")
        
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


def main_qa_system(clips_data, llm_infer_function, user_question, database_json_path):
    """
    Main QA system that answers user questions about daily life using audio recordings.
    
    Args:
        clips_data (list): Audio clips database
        llm_infer_function (function): LLM inference function
        user_question (str): User's question (if None, will prompt for input)
        
    Returns:
        str: Final answer to user's question
    """
    
    print(f"{AudioDaily_normal} Processing...")
    
    # Step 1: Get time chunks covered in database
    available_time_chunks = parse_time(clips_data)
    print(f"{AudioDaily_normal} Available time chunks in database: {len(available_time_chunks)} chunks")
    
    # Step 2: Ask LLM which time chunks are needed to answer the question
    # Run the LLM call in a separate process and retrieve the result via a Queue
    ctx = get_context('fork')  # use 'fork' to avoid pickling issues on Unix-like systems
    result_queue = ctx.Queue()

    def _time_chunks_worker(avail_chunks, question, llm_fn, out_q):
        try:
            res = ask_llm_for_relevant_time_chunks(avail_chunks, question, llm_fn)
            out_q.put({'ok': True, 'result': res})
        except Exception as e:
            out_q.put({'ok': False, 'error': str(e)})

    proc = ctx.Process(target=_time_chunks_worker, args=(available_time_chunks, user_question, llm_infer_function, result_queue))
    proc.start()

    # Wait for result with timeout and ensure process ends
    TIMEOUT_SECONDS = 900
    proc.join(TIMEOUT_SECONDS)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        print(f"{AudioDaily_error} LLM time-chunk process timed out and was terminated.")
        needed_time_chunks = []
    else:
        try:
            out = result_queue.get_nowait()
            if out.get('ok'):
                needed_time_chunks = out.get('result', [])
            else:
                print(f"{AudioDaily_error} Error from LLM process: {out.get('error')}")
                needed_time_chunks = []
        except Exception as e:
            print(f"{AudioDaily_error} Failed to retrieve result from LLM process: {e}")
            needed_time_chunks = []
            
    # needed_time_chunks = ask_llm_for_relevant_time_chunks(
    #     available_time_chunks, user_question, llm_infer_function
    # )
    
    if not needed_time_chunks:
        return f"{AudioDaily_error} I couldn't find relevant time periods in the available data to answer your question."
    
    print(f"{AudioDaily_normal} LLM identified {len(needed_time_chunks)} relevant time chunks")
    
    # Step 3: Retrieve clips within the needed time chunks
    relevant_clips = retrieve_time(clips_data, needed_time_chunks)
    print(f"{AudioDaily_normal} Retrieved {len(relevant_clips)} relevant audio clips")
    
    # Step 4: Ask LLM if it can answer with label1 information
    # Run initial analysis in a separate process and retrieve result via a Queue
    init_result_queue = ctx.Queue()

    def _initial_analysis_worker(clips, question, llm_fn, out_q):
        try:
            res = ask_llm_initial_analysis(clips, question, llm_fn)
            out_q.put({'ok': True, 'result': res})
        except Exception as e:
            out_q.put({'ok': False, 'error': str(e)})

    proc2 = ctx.Process(
        target=_initial_analysis_worker,
        args=(relevant_clips, user_question, llm_infer_function, init_result_queue)
    )
    proc2.start()

    proc2.join(TIMEOUT_SECONDS)
    if proc2.is_alive():
        proc2.terminate()
        proc2.join()
        print(f"{AudioDaily_error} LLM initial-analysis process timed out and was terminated.")
        initial_response = ""
    else:
        try:
            out = init_result_queue.get_nowait()
            if out.get('ok'):
                initial_response = out.get('result', "")
            else:
                print(f"{AudioDaily_error} Error from LLM initial-analysis process: {out.get('error')}")
                initial_response = ""
        except Exception as e:
            print(f"{AudioDaily_error} Failed to retrieve result from LLM initial-analysis process: {e}")
            initial_response = ""
    
    # initial_response = ask_llm_initial_analysis(relevant_clips, user_question, llm_infer_function)
    
    # Step 5: Check LLM response
    if initial_response.startswith("ANSWER_FOUND"):
        # Extract and return the answer
        answer = extract_answer_from_response(initial_response)
        return f"{AudioDaily_normal} Answer: {answer}"
    
    elif initial_response.startswith("ANSWER_NOT_FOUND"):
        # Extract timestamps that need more detail
        needed_timestamps = extract_timestamps_from_response(initial_response)
        print(f"{AudioDaily_normal} LLM needs more detailed information for {len(needed_timestamps)} clips")
        
        # Step 6: Generate label2 for the needed clips
        # Run detailed labeling in a separate process and retrieve results
        label_result_queue = ctx.Queue()

        def _label_worker(clips, timestamps, llm_fn, out_q):
            try:
                res = label_clips_2(clips, timestamps, llm_fn, database_json_path)
                out_q.put({'ok': True, 'result': res})
            except Exception as e:
                out_q.put({'ok': False, 'error': str(e)})

        proc3 = ctx.Process(
            target=_label_worker,
            args=(clips_data, needed_timestamps, llm_infer_function, label_result_queue)
        )
        proc3.start()

        proc3.join(TIMEOUT_SECONDS)
        if proc3.is_alive():
            proc3.terminate()
            proc3.join()
            print(f"{AudioDaily_error} LLM detailed-label process timed out and was terminated.")
            updated_clips_data = clips_data
        else:
            try:
                out = label_result_queue.get_nowait()
                if out.get('ok'):
                    updated_clips_data = out.get('result', clips_data)
                else:
                    print(f"{AudioDaily_error} Error from detailed-label process: {out.get('error')}")
                    updated_clips_data = clips_data
            except Exception as e:
                print(f"{AudioDaily_error} Failed to retrieve result from detailed-label process: {e}")
                updated_clips_data = clips_data
        
        # updated_clips_data = label_clips_2(clips_data, needed_timestamps, llm_infer_function)
        
        # Step 7: Get final answer with detailed label2 information
        # Step 7: Run final LLM analysis in a separate process and retrieve result via a Queue
        final_result_queue = ctx.Queue()

        def _final_analysis_worker(clips, timestamps, question, llm_fn, out_q):
            try:
                res = ask_llm_final_analysis(clips, timestamps, question, llm_fn)
                out_q.put({'ok': True, 'result': res})
            except Exception as e:
                out_q.put({'ok': False, 'error': str(e)})

        proc4 = ctx.Process(
            target=_final_analysis_worker,
            args=(updated_clips_data, needed_timestamps, user_question, llm_infer_function, final_result_queue)
        )
        proc4.start()

        proc4.join(TIMEOUT_SECONDS)
        if proc4.is_alive():
            proc4.terminate()
            proc4.join()
            print(f"{AudioDaily_error} LLM final-analysis process timed out and was terminated.")
            final_answer = "The final LLM analysis timed out."
        else:
            try:
                out = final_result_queue.get_nowait()
                if out.get('ok'):
                    final_answer = out.get('result', "")
                else:
                    print(f"{AudioDaily_error} Error from final-analysis process: {out.get('error')}")
                    final_answer = "An error occurred during final LLM analysis."
            except Exception as e:
                print(f"{AudioDaily_error} Failed to retrieve result from final-analysis process: {e}")
                final_answer = "Failed to retrieve final LLM analysis result."
        
        # final_answer = ask_llm_final_analysis(
        #     updated_clips_data, needed_timestamps, user_question, llm_infer_function
        # )
        return f"{AudioDaily_normal} Answer: {final_answer}"
    
    else:
        return f"{AudioDaily_error} I encountered an issue while processing your question. Please try again."


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
1. The returned time chunk in the list should be either within one of the existing chunks or contains parts of them
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
        print(f"{AudioDaily_error} Error parsing LLM response for time chunks: {e}")
        return []


def retrieve_time(clips_data, time_chunks):
    """
    Retrieve clips that fall within the specified time chunks.
    
    Args:
        clips_data (list): List of clip dictionaries from the JSON database
        time_chunks (list): List of time chunks in format "YYYY/MM/DD HH:MM:SS-YYYY/MM/DD HH:MM:SS"
        
    Returns:
        list: List of clip dictionaries that fall within the specified time chunks
    """
    if not clips_data or not time_chunks:
        print(f"{AudioDaily_error} No clips data or time chunks provided.")
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



if __name__ == "__main__":
    database_json_path = "/aiot-nvme-15T-x2-hk01/home/aiot25-suyihang/AudioDaily/test_results/metadata.json"
    # metadata = chunk_audio_clips("/aiot-nvme-15T-x2-hk01/home/aiot25-suyihang/AudioDaily/test_audios/251101_180000.mp3",
    #                              database_json_path,
    #                              "/aiot-nvme-15T-x2-hk01/home/aiot25-suyihang/AudioDaily/test_results")
    # labeled_metadata_1 = label_clips_1(metadata, database_json_path, qwen3o_infer)



    # Load your clips data
    try:
        with open(database_json_path, 'r') as f:
            clips_data = json.load(f)
    except FileNotFoundError:
        print(f"{AudioDaily_error} Error: audio_clips_database.json not found. Please process your audio files first.")
        exit(1)

    # Run the QA system
    print(f"{AudioDaily_normal} === AudioDaily: Daily Life Audio QA System ===")
    print(f"{AudioDaily_normal} This system answers questions about your daily life using audio recordings from your wearable device.")

    while True:
        user_question = input(f"\n{AudioDaily_normal} Enter your question (or 'quit' to exit): ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            break
            
        if not user_question:
            continue
            
        user_question = "Today is " + datetime.now().strftime("%Y/%m/%d") + ". " + user_question
        answer = main_qa_system(clips_data, qwen3o_infer, user_question, database_json_path)
        print(f"\n{answer}\n")
        
    print(f"{AudioDaily_normal} Exiting AudioDaily QA System. Goodbye!")
    sys.exit(0)
