import streamlit as st
import subprocess
from pydub import AudioSegment
import math
import glob
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


llm = ChatOpenAI(
    temperature=0.1,
)

has_transcript = os.path.exists("./.cache/eng_vedio.txt")

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size = 1000,
                chunk_overlap = 200,
            )



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
            text_file.write(f'{transcript["text"]} \n')

@st.cache_data(show_spinner="Embedding text...")
def embed_file(file_path, file_name):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file_name}")
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embedding = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embedding, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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
    with st.status("Uploading video file...") as status:
        video_content = video.read()
        file_name = video.name
        video_path = f"./.cache/{video.name}"
        format = video_path[-3:]
        audio_path = video_path.replace(format,"mp3")
        transcript_path = video_path.replace(format,"txt")
        with open(video_path,"wb") as f:
            f.write(video_content)
    
        status.update(label="Extracting audio file...")
        extract_audio_from_video(video_path, audio_path)
    
        status.update(label="Cutting audio...")
        cut_audio_in_chunks(audio_path,10,chunk_folder)
        status.update(label="Transcribing audio...")
        transcribe_chunks(chunk_folder, transcript_path)

    transcript, summary, qa = st.tabs(["Transcript","Summary","Q&A"])

    with transcript:
        with open(transcript_path, "r") as file:
            st.write(file.read())
    
    with summary:
        start = st.button("요약본 생성하기 📄")

        if start:

            loader = TextLoader(transcript_path)

            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size = 1000,
                chunk_overlap = 200,
            )

            docs = loader.load_and_split(text_splitter=splitter)
            
            first_prompt = ChatPromptTemplate.from_template(
                """
            Write a concise summary of the following:
            "{text}"
            CONCISE SUMMARY :
            """
            )

            first_chain = first_prompt | llm | StrOutputParser()

            summary = first_chain.invoke({"text": docs[0].page_content})

            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up to a certain point: {existing_summary}
                We have the opportunity to refine the existing summary (only if needed) with some more context below.
                ------------
                {context}
                ------------
                Given the new context, refine the original summary.
                If the context isn't useful, RETURN the original summary.
                """
            )

            refine_chain = refine_prompt | llm | StrOutputParser()

            with st.status("Summarizing...✍️") as status:
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document {i+1}/{len(docs)-1}")
                    summary = refine_chain.invoke({"existing_summary":summary,
                                                   "context":doc.page_content},)
                    
                    st.write(summary)
            st.markdown("##### Final Summary ! 👇")
            st.write(summary)


    with qa:

        retriever = embed_file(transcript_path, file_name)
        # doc = retriever.invoke("do they talk about marcus aurelius?")
        # st.write(doc)

        send_message("질문을 기다리고 있습니다 🤖", "ai", save=False)
        paint_history()
        message = st.chat_input("비디오 파일 내용에 대해 무엇이든 물어보세요 🙆‍♀️!")

        if message:
            send_message(message, "human")
            qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
            """
            주어진 context만을 사용하여 질문에 답변하세요. 만약 답을 잘 모르겠다면, '정확한 답을 찾기 어렵습니다. 
            다른 질문을 해보시겠어요?' 라고 답변하세요. 절대 답변을 꾸며내지 마세요.

            Context: {context}
            """),
            ("human","{question}"),
            ])

            qa_chain = {"context": retriever|RunnableLambda(format_docs),
                        "question": RunnablePassthrough()} | qa_prompt | llm | StrOutputParser()
            

            response = qa_chain.invoke(message)

            send_message(response, "ai")

else:
    st.session_state["messages"] = []


            
