import os
import requests
import shutil
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from processor import (
    load_whisper,
    create_srt,
    analyze_with_gemini,
    process_ffmpeg_pipeline,
    create_short,
    BASE_DIR,
    UPLOAD_FOLDER,
    OUTPUT_FOLDER,
    LOGO_PATH
)

app = FastAPI()

# Mount outputs directory to serve files
app.mount("/outputs", StaticFiles(directory=OUTPUT_FOLDER), name="outputs")

class ProcessRequest(BaseModel):
    video_url: str
    gemini_api_key: str
    webhook_url: str = None  # Optional: Call this URL when done (for async processing)

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

@app.post("/process")
async def process_video(request: ProcessRequest):
    try:
        # 1. Setup
        client = genai.Client(api_key=request.gemini_api_key)
        whisper_model = load_whisper()
        
        # 2. Download Video
        # Extract filename from URL or use default
        filename = request.video_url.split("/")[-1].split("?")[0]
        if not filename.endswith(('.mp4', '.mov')):
            filename = "downloaded_video.mp4"
            
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        download_file(request.video_url, input_path)
        
        base_name = os.path.splitext(filename)[0]
        
        # 3. Transcribe
        transcription = whisper_model.transcribe(input_path)
        transcript_text = transcription["text"]
        
        srt_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.srt")
        create_srt(transcription, srt_path)
        
        # 4. Analyze
        metadata = analyze_with_gemini(client, transcript_text)
        if not metadata:
            raise HTTPException(status_code=500, detail="Gemini analysis failed")
            
        # 5. Process Video (Master)
        master_filename = f"master_{filename}"
        master_path = os.path.join(OUTPUT_FOLDER, master_filename)
        
        process_ffmpeg_pipeline(input_path, master_path, srt_path, LOGO_PATH)
        
        # 6. Process Short
        short_filename = f"short_{base_name}.mp4"
        short_path = os.path.join(OUTPUT_FOLDER, short_filename)
        
        v_start = metadata['viral_segment']['start']
        v_end = metadata['viral_segment']['end']
        create_short(input_path, short_path, v_start, v_end)
        
        # 7. Construct Response
        # Assuming the server is reachable at the host header, but for now returning relative paths
        # In a real scenario, you'd want the full public URL (e.g. from ngrok)
        
        return {
            "status": "success",
            "description": metadata['description'],
            "thumbnail_prompt": metadata['thumbnail_prompt'],
            "thumbnail_text_overlay": metadata.get('thumbnail_text_overlay', ''),
            "master_video_path": f"/outputs/{master_filename}",
            "short_video_path": f"/outputs/{short_filename}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
