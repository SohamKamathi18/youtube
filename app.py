import streamlit as st
import os
import subprocess
import whisper
import json
from datetime import timedelta
from google import genai
from google.genai import types
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

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Video Studio", layout="wide")

# --- SIDEBAR & API SETUP ---
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Google Gemini API Key", type="password")

# --- MAIN UI LAYOUT ---
st.title("🎥 AI Content Factory")
st.markdown("Upload a raw video → Get a **Mastered Video**, **Viral Short**, **Thumbnail Prompt**, and **Description**.")

uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov'])

if uploaded_file and api_key:
    if st.button("Start AI Pipeline"):
        
        # 1. Setup
        client = genai.Client(api_key=api_key)
        whisper_model = load_whisper()
        
        filename = uploaded_file.name.replace(" ", "_")
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        base_name = os.path.splitext(filename)[0]
        
        # --- PIPELINE STATUS ---
        with st.status("Running Pipeline...", expanded=True) as status:
            
            # Step A: Transcribe
            st.write("📝 Transcribing audio...")
            transcription = whisper_model.transcribe(input_path)
            transcript_text = transcription["text"]
            
            srt_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.srt")
            create_srt(transcription, srt_path)
            
            # Step B: Gemini Intelligence
            st.write("🧠 Analyzing content with Gemini...")
            metadata = analyze_with_gemini(client, transcript_text)
            
            if metadata:
                # Step C: Video Processing
                st.write("⚙️ FFmpeg: Mastering Audio & Burning Subtitles...")
                master_path = os.path.join(OUTPUT_FOLDER, f"master_{filename}")
                
                # Ensure a logo exists, or use a placeholder logic if needed
                if not os.path.exists(LOGO_PATH):
                    st.warning("⚠️ No logo.png found in static folder! Skipping logo overlay.")
                
                try:
                    process_ffmpeg_pipeline(input_path, master_path, srt_path, LOGO_PATH)
                    
                    st.write("✂️ FFmpeg: Creating Viral Vertical Short...")
                    short_path = os.path.join(OUTPUT_FOLDER, f"short_{base_name}.mp4")
                    v_start = metadata['viral_segment']['start']
                    v_end = metadata['viral_segment']['end']
                    create_short(input_path, short_path, v_start, v_end)
                    
                    status.update(label="Pipeline Complete!", state="complete", expanded=False)
                    
                    # --- DISPLAY RESULTS ---
                    st.divider()
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Mastered Video")
                        st.video(master_path)
                        with open(master_path, 'rb') as f:
                            st.download_button("Download Master", f, file_name=f"master_{filename}")

                    with col2:
                        st.subheader("Viral Short (9:16)")
                        st.video(short_path)
                        with open(short_path, 'rb') as f:
                            st.download_button("Download Short", f, file_name=f"short_{filename}")
                            
                    st.divider()
                    
                    col3, col4 = st.columns([1, 2])
                    with col3:
                        st.subheader("Thumbnail Idea")
                        st.info("Copy the prompt below to generate a thumbnail.")
                        
                        full_prompt = f"IMAGE PROMPT: {metadata['thumbnail_prompt']}\n\nTEXT OVERLAY: {metadata.get('thumbnail_text_overlay', 'N/A')}"
                        st.code(full_prompt, language="text")
                    
                    with col4:
                        st.subheader("YouTube Details")
                        st.text_area("Description", value=metadata['description'], height=200)
                        
                except subprocess.CalledProcessError as e:
                    st.error(f"FFmpeg Error: {e}")
            else:
                st.error("Failed to generate metadata from Gemini.")

elif not api_key:
    st.warning("Please enter your Google Gemini API Key in the sidebar.")