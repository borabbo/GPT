import streamlit as st
import time

from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings, OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS 
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory


st.set_page_config(
    page_title="Private GPT",
    page_icon="🤫"
)

class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self, *args, **kwargs):
        self.tokens = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
        self.tokens = ""
    
    def on_llm_end(self, *args, **kwargs):
        save_message(self.tokens, "ai")
    
    def on_llm_new_token(self, token:str, *args, **kwargs):
        self.tokens += token
        self.message_box.markdown(self.tokens)

# llm = ChatOllama(
#    model="mistral:latest",
#    temperature=0.1,
#    streaming=True,
#    callbacks=[ChatCallbackHandler(),],
# )

with st.sidebar:
    model = st.selectbox("Choose Your model", ("mistral","llama2"))
    if model == "mistral":
        llm = ChatOllama(
            model="mistral:latest",
            temperature=0.1, 
            streaming=True,
            callbacks=[ChatCallbackHandler(),
               ],
               )
    else:
        llm = ChatOllama(
            model="llama2:latest",
            temperature=0.1,
            streaming=True,
            callbacks=[ChatCallbackHandler(),],
        )


st.title("Private GPT 🤫")


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,

    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OllamaEmbeddings(model="mistral:latest")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstors = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstors.as_retriever()

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


prompt = ChatPromptTemplate.from_template(
     """
    Answer the question using ONLY the following context and not your
    training data. If you don't know the answer just say you don't know.
    DON'T make anything up.

    Context : {context} 
    Question: {question}
    """
) 

st.markdown("""
Welcome !

너무 긴 문서를 전부 읽고 내용을 확인하기 어려우시죠?
            
여기 BORA GPT WORLD의 Document GPT를 활용하세요 😉

파일을 업로드하고, 관련된 내용에 대해 무엇이든 물어보세요 🔍
            
""")

with st.sidebar:
    file = st.file_uploader(label="파일을 업로드하세요!*가능한 확장자= .txt .docx .pdf",
                            type=["pdf","txt","docx"])


if file:
    retriever = embed_file(file)

    send_message("파일이 준비되었습니다 ! 무엇이든 물어보세요 🙆‍♀️", "ai",False)
    paint_history()
    message = st.chat_input("업로드한 문서의 내용과 관련하여 무엇이든 질문해보세요 😉")

    if message:
        send_message(message, "human")
        chain = {"context": retriever | RunnableLambda(format_docs),
                 "question":RunnablePassthrough()} | prompt | llm
        
        with st.chat_message("ai"):
            chain.invoke(message)

else:
    st.session_state["messages"] = []        

