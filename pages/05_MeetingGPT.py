import streamlit as st
import subprocess
from pydub import AudioSegment
import math
import glob
import os
import openai

has_transcript = os.path.exists("./.cache/eng_vedio.txt")

@st.cache_data()
def extract_audio_from_video(video_path, audio_path):
    if has_transcript:
        return
    command = ["/opt/homebrew/bin/ffmpeg", "-i", video_path, "-vn", audio_path, "-y"]
    subprocess.run(command)

@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track)/chunk_len)
 
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/{i}_chunk.mp3", format="mp3")

@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return
    
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
            )
            text_file.write(transcript["text"])

st.set_page_config(
    page_title="MeetingGPT",
    page_icon="🎙️",
)

st.markdown(
    """
# MeetingGPT

## Welcome Meeting GPT !🙆‍♀️ \n
비디오 파일을 업로드 하세요 ! \n
그럼 텍스트로 된 스크립트를 받을 수 있어요 \n
비디오 내용 요약본을 받거나, 비디오에 관련된 질문을 하면 답변해 드립니다 😎

##### 좌측 화면에서 파일을 업로드하세요 ! Let's Start 🤖
"""
)

with st.sidebar:
    video = st.file_uploader("video", type=[".mp4","mov","avi"])

if video:
    chunk_folder = "./.cache/chunks"
    with st.status("Uploading video file..."):
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4","mp3")
        transcript_path = video_path.replace("mp4","txt")
        with open(video_path,"wb") as f:
            f.write(video_content)
    
    with st.status("Extracting audio file..."):
        extract_audio_from_video(video_path, audio_path)
    
    with st.status("Cutting audio..."):
        cut_audio_in_chunks(audio_path,10,chunk_folder)
    with st.status("Transcribing audio..."):
        transcribe_chunks(chunk_folder, transcript_path)
