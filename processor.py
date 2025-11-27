import os
import subprocess
import json
import whisper
from datetime import timedelta
from google import genai
from google.genai import types

# Directory Setup (Replicated here for standalone usage)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'temp_uploads')
LOGO_PATH = os.path.join(BASE_DIR, 'static', 'Logo_of_YouTube_(2015-2017).png')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_whisper():
    """Load model once and cache it."""
    return whisper.load_model("base")

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def create_srt(transcription_result, srt_path):
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(transcription_result["segments"], start=1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    return srt_path

def analyze_with_gemini(client, transcript_text):
    prompt = f"""
    Analyze this video transcript:
    "{transcript_text[:8000]}..."

    Return a strict JSON object with these keys:
    1. "description": A catchy YouTube description.
    2. "thumbnail_prompt": A highly visual description for an image generator (robot, cyberpunk, etc).
    3. "thumbnail_text_overlay": Short, punchy text to place on the thumbnail (max 5 words).
    4. "viral_segment": {{ "start": int_seconds, "end": int_seconds }} for the best 30s clip.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', # Use a capable text model
            contents=prompt
        )
        clean_json = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_json)
    except Exception as e:
        print(f"Gemini Analysis Error: {e}")
        return None

def process_ffmpeg_pipeline(input_path, output_path, srt_path, logo_path=None):
    # Windows/Linux path handling for FFmpeg
    srt_path_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
    
    inputs = ['-i', input_path]
    
    if logo_path and os.path.exists(logo_path):
        inputs.extend(['-i', logo_path])
        # 1. Overlay Logo (Bottom Right, Scaled to 150px width)
        # 2. Burn Subtitles (Black Text, White Box, Small Font)
        # 3. Normalize Audio (Loudnorm)
        filter_complex = (
            f"[1:v]scale=150:-1[v_logo_scaled];"
            f"[0:v][v_logo_scaled]overlay=main_w-overlay_w-20:main_h-overlay_h-20[v_logo];"
            f"[v_logo]subtitles='{srt_path_escaped}':force_style='Fontname=Arial,FontSize=12,PrimaryColour=&H000000,BackColour=&HFFFFFF,BorderStyle=3,Outline=0,MarginV=20'[v_out];"
            f"[0:a]loudnorm=I=-16:TP=-1.5:LRA=11[a_out]"
        )
    else:
        # No logo, just subtitles and audio norm
        filter_complex = (
            f"[0:v]subtitles='{srt_path_escaped}':force_style='Fontname=Arial,FontSize=12,PrimaryColour=&H000000,BackColour=&HFFFFFF,BorderStyle=3,Outline=0,MarginV=20'[v_out];"
            f"[0:a]loudnorm=I=-16:TP=-1.5:LRA=11[a_out]"
        )

    command = [
        'ffmpeg', '-y',
        *inputs,
        '-filter_complex', filter_complex,
        '-map', '[v_out]', '-map', '[a_out]',
        output_path
    ]
    subprocess.run(command, check=True)

def create_short(input_path, output_path, start, end):
    duration = end - start
    # Scale to fit 1080x1920 (9:16) with letterboxing (Black Bars)
    # This ensures the full horizontal video is visible inside the vertical frame
    command = [
        'ffmpeg', '-y',
        '-ss', str(start), '-t', str(duration),
        '-i', input_path,
        '-filter:v', "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
        '-c:a', 'copy',
        output_path
    ]
    subprocess.run(command, check=True)
