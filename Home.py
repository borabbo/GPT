import streamlit as st

from datetime import datetime

today = datetime.today().strftime("%H:%M:%S")

#st.title("BORA's GPT World !!! 💜")
#st.subheader(today)

st.set_page_config(
    page_title="Bora's GPT Home",
    page_icon="🤖"
)

st.markdown(
    """
# Bora's GPT World !!! 💜 🙆‍♀️

Welcome to Bora's GPT Portfolio ❣️

Here are the apps I made:

- [ ] [DocumentGPT](/DocumentGPT)
- [ ] [PrivateGPT](/PrivateGPT)
- [ ] [QuizGPT](/QuizGPT)
- [ ] [SiteGPT](/SiteGPT)
- [ ] [MeetingGPT](/MeetingGPT)
- [ ] [InvestorGPT](/InvestorGPT)
"""
)






#selectbox 선택에 따라 옵션기능 숨기기 

#model = st.selectbox("Choose your option",("GPT-3","GPT_4"))
#if model == "GPT-3":
#    st.write("Cheap")

#else:
#    st.write("Not cheap")
#    name = st.text_input("What is your name?")
#    st.write(name)

#    value = st.slider("temperature",min_value=0.1, max_value=1.0)
#    st.write(value)

# 지도 삽입하기

#m = folium.Map(location=[37.21101, 127.1021], zoom_start=16)
#folium.Marker(
#    [37.21101, 127.1021],
#    popup="우리집",
#    tooltip="사랑이넘치는우리집"
#).add_to(m)
#st_data = st_folium(m, width=200, height=200)

