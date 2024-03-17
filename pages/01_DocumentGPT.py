from typing import Any, Dict, List, Optional
from uuid import UUID
import streamlit as st
import time
#from langchain_community.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
#from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredFileLoader
#from langchain_community.embeddings import OpenAIEmbeddings
#from langchain.embeddings import CacheBackedEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
#from langchain_community.vectorstores.faiss import FAISS
from langchain.vectorstores import FAISS 
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📑",
)

@st.cache_resource
def init_llm(chat_callback: bool):
    if chat_callback == True:

        class ChatCallbackHandler(BaseCallbackHandler):
            def __init__(self, *args, **kwargs):
                self.tokens = ""
            
            def on_llm_start(self,*args, **kwargs):
                self.message_box = st.empty()
                self.tokens = ""

            def on_llm_end(self,*args, **kwargs):
                save_message(self.tokens, "ai")

            def on_llm_new_token(self, token:str, *args, **kwargs):
                self.tokens += token
                self.message_box.markdown(self.tokens)
                #with self.messagebox:
                #    st.write(self.tokens) 
    
        callback = [ChatCallbackHandler()]
    
    else:
        callback = []
    
    return ChatOpenAI(
        temperature=0.1, streaming=True, callbacks=callback
    )
    


llm_for_chat = init_llm(chat_callback=True)
llm_for_memory = init_llm(chat_callback=False)

@st.cache_resource
def init_memory(_llm):
    return ConversationSummaryBufferMemory(llm=llm_for_memory, max_token_limit=200, return_messages=True, memory_key="history")


memory = init_memory(llm_for_memory)



st.title("Document GPT 📑")



@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings,
                                                               cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role":role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

@st.cache_resource
def load_memory(_):
    return memory.load_memory_variables({})["history"]

#def save_memory():
#    memory_history = memory.load_memory_variables({})["history"]
#    st.session_state["memories"].append(memory_history)

#def invoke_chain(question):
#    result = chain.invoke(question)
#    memory.save_context({"inputs":question},{"outputs":result.content})
#    return result



prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
    주어진 context만을 사용하여 질문에 답변하세요. 만약 답을 잘 모르겠다면, '정확한 답을 찾기 어렵습니다. 
    다른 질문을 해보시겠어요?' 라고 답변하세요. 절대 답변을 꾸며내지 마세요.

    Context: {context}
    """),
    MessagesPlaceholder(variable_name="history"),
    ("human","{question}"),
])


st.markdown("""
Welcome !

너무 긴 문서를 전부 읽고 내용을 확인하기 어려우시죠?
            
여기 BORA GPT WORLD의 Document GPT를 활용하세요 😉

파일을 업로드하고, 관련된 내용에 대해 무엇이든 물어보세요 🔍
            
""")

with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file",
                            type=["pdf","txt","docx"])

if file:
    retriever = embed_file(file)

    send_message("파일이 준비되었습니다. 무엇이든 물어보세요!", "ai", save=False)
    paint_history()
    message = st.chat_input("업로드한 문서의 내용 중 무엇이든 물어보세요 🙆‍♀️")

    if message:
        send_message(message, "human")
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            "history": load_memory,
        } | prompt | llm_for_chat

        with st.chat_message("ai"):
            response = chain.invoke(message).content
            memory.save_context({"inputs": message}, {"output": response})
        #send_message(response.content, "ai")
      

else:

    st.session_state["messages"] = [] 
    memory.clear()

