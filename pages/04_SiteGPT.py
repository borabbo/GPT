import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks.base import BaseCallbackHandler



st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️"
)

@st.cache_resource
def init_llm(chat_callback: bool):
    if chat_callback:
        class ChatCallbackHandler(BaseCallbackHandler):
            def __init__(self, *args, **kwars):
                self.tokens = ""
            
            def on_llm_start(self, *args, **kwargs):
                self.message_box = st.empty()
                self.tokens = ""
            
            def on_llm_end(self, *args, **kwargs):
                save_message(self.tokens, "ai")
            
            def on_llm_new_token(self, token:str, *args, **kwargs):
                self.tokens += token
                self.message_box.markdown(self.tokens)
        
        callback = [ChatCallbackHandler()]
    else:
        callback = []
    
    return ChatOpenAI(temperature=0.1, streaming=True, callbacks= callback)


llm_for_memory = init_llm(chat_callback=False)
llm_for_chat = init_llm(chat_callback=True)

llm = ChatOpenAI(temperature=0.1,)


@st.cache_resource
def init_memory(_llm):
    return ConversationSummaryBufferMemory(llm=_llm, max_token_limit=400, 
                                           return_messages=True, memory_key="history")

memory = init_memory(llm_for_memory)

@st.cache_resource
def load_memory(_):
    return memory.load_memory_variables({})["history"]



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


answers_prompt = ChatPromptTemplate.from_template(
"""
Using ONLY the following context answer the user's question.
If you can't just say you don't know. Don't make anything up.

Then, give a score to the answer between 0 and 5.                                                                                                   
The score should be high if the answer is related to the user's question.
and low otherwise.
If there is no relevant content, the score is 0.
If there is no information provided, the score is 0.
MAKE SURE to ALWAYS include the answer's score even if it's 0.

Context: {context}

Examples:
Question: How far away is the moon?
Answer: The moon is 384,400 km away.
Score: 5

Question: How far away is the sun?
Answer: I don't know
Score: 0

Your turn !

Question: {question}

                                                                                                                                                                                                
""")

def get_answers(inputs):
    docs = inputs['docs']
    question = inputs['question']
    answers_chain = answers_prompt | llm

    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke({
    #         "question": question,
    #         "context": doc.page_content
    #     })
    #     answers.append(result.content)
    return {
        "question": question,
        "answers":[
        {
           "answer": answers_chain.invoke(
               {"question":question, "context":doc.page_content}
           ).content,
           "source": doc.metadata["source"],
           "date":doc.metadata["lastmod"],
        } for doc in docs],
       }

choose_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
    Use ONLY the following pre-existing answers to answer the user's question.

    Use the answers that have the highest score (more helpful) and
    favor the most recent ones.

    Return the sources of the answers as they are, DO NOT change them.

    Answers: {answers}
    """),
    MessagesPlaceholder(variable_name="history"),
    ("human","{question}")
]
)

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = RunnablePassthrough.assign(history=load_memory) | choose_prompt | llm_for_chat
    condensed = "\n\n".join(f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
                            for answer in answers)
    return choose_chain.invoke({"question": question, "answers":condensed})

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()   
    return str(soup.get_text()).replace("\n"," ").replace("\xa0"," ").replace("CloseSearch Submit Blog","")



@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200,)
    loader = SitemapLoader(url,
                        #    filter_urls=[
                        #        r"^(.*\/blog\/).*",#부정기호:?! = "^(?!.*\/blog\/).*"
                        #        #\: 문자 그대로를 의미하는 정규표현식 = /blog 와 / 를 의미
                        #    ],
                           parsing_function=parse_page)
    loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()



html2text_t = Html2TextTransformer()

st.title("Site GPT")

st.markdown("""
Welecome to Site GPT 🙆‍♀️

""")


with st.sidebar:
    url = st.text_input("URL을 입력하세요", placeholder="hhtps://www.example.com")

if url:
    #st.session_state["messages"] = []
    #memory.clear()
    # #async chromium loader
    #loader = AsyncChromiumLoader([url])
    # docs = loader.load()
    # transformed = html2text_t.transform_documents(docs)
    # st.write(docs)
    if ".xml" not in url:
        with st.sidebar:
            st.error("url 주소를 확인하세요. '.xml' 주소가 필요합니다") #/sitemap.xml
    else:
        retriever = load_website(url)
        # result = retriever.invoke("What is the pricing of GPT-4 Turbo with vision")
        # st.write(result)

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        send_message("입력하신 URL의 웹사이트에서 모든 정보를 준비하였습니다 ", "ai",False)
        paint_history()
        
        message = st.chat_input("웹사이트에서 필요한 정보를 무엇이든 물어보세요 😉")
        if message:
            send_message(message, "human")

            chain = {"docs": retriever, 
                    "question": RunnablePassthrough(),
                    } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)
            
            with st.chat_message("ai"):
                result = chain.invoke(message).content
                memory.save_context({"inputs":message}, {"outputs": result})
            st.write(memory.load_memory_variables({})["history"])
        # else:
        #     st.session_state["messages"] = []
        #     memory.clear()
        #     st.session_state["messages"] = []

else:
    st.session_state["messages"] = []
    memory.clear()

        