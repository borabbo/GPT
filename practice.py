import streamlit as st
from streamlit_folium import st_folium
import folium

from datetime import datetime
import time

today = datetime.today().strftime("%H:%M:%S")

st.title("BORA's Practice")
st.subheader(today)

with st.sidebar:
    st.title("GPT World")
    st.title("Practice")
    st.text_input("title")

tab_one, tab_two, tab_three = st.tabs(["지도","Select Box","C"])

with tab_one:
    st.write("우리집 지도입니다")
    # 지도 삽입하기

    m = folium.Map(location=[37.21101, 127.1021], zoom_start=16)
    folium.Marker(
        [37.21101, 127.1021],
        popup="우리집",
        tooltip="사랑이넘치는우리집"
        ).add_to(m)
    st_data = st_folium(m, width=200, height=200)

with tab_two:
    st.write("Select Box 기능입니다")
    model = st.selectbox("Choose Your Model", ("GPT-3","GPT-4"))
    if model == "GPT-3":
        st.write("Cheap Model")
    else:
        st.write("Not Cheap 🥲")
        name = st.text_input("What is your name ?")
        st.write(name)

        value = st.slider("temperature", min_value=0.1, max_value=1.0)
        st.write(value)




#with st.status("Embedidng file...", expanded=True) as status:
#    time.sleep(2)
#    st.write("Getting the file")
#    time.sleep(2)
#    st.write("Embedding the file")
#    time.sleep(2)
#    st.write("Caching the file")
#    status.update(label="Error", state="error")


st.title("Document GPT 📑")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message":message, "role":role})

for message in st.session_state["messages"]:
    send_message(message["message"], message["role"], save=False)

message = st.chat_input("Send a message to the ai")


if message:
    send_message(message, "human")
    time.sleep(2)
    send_message(f"You said : {message}", "ai")

    #Sitebar에 저장데이터 표기하기 :
with st.sidebar:
    st.write(st.session_state)


