
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
import json

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```","").replace("json","")#.replace(",]","]").replace(",}","}")
        return json.loads(text)
    

output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="🧐",
)

st.title("QuizGPT")


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler()
    ],
)

quiz_function = {
    "name":"create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions":{
                "type": "array",
                "items":{
                    "type": "object",
                    "properties":{
                        "question": {
                            "type": "string"},
                        "answers": {
                            "type": "array",
                            "items":{
                                "type": "object",
                                "properties":{
                                    "answer":{"type":"string"},
                                    "correct": {"type": "boolean"}
                                }
                            },
                            "required": ["answer", "correct"],
                        }
                    }
                },
                "required": ["question", "answers"],
            }
        },
        "required": ["questions"],
    }

}

func_llm = ChatOpenAI(
    temperature=0.1, 
    model = "gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
).bind(function_call={"name":"create_quiz"},
       functions = [quiz_function])




@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap = 100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs
  

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

@st.cache_data(show_spinner="Making quiz...")
def func_quiz_chain(_docs, topic):
    chain = {"context": format_docs} | func_prompt | func_llm
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context":question_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)

@st.cache_data(show_spinner="Search wikipidia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


func_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
        You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
    questions have to be 10 questions.
    each answers have four answer. one correct answer and three false answer
    if context language is Korean, you should make it by Korean.

    Context : {context}
""")
    ]
)

question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
    if context language is Korean, you should make it by Korean.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: 한국의 수도는 어디입니까?
    Answers: 서울(o) | 부산 | 대구 | 제주

    Question: 비빔밥은 어느 나라 음식입니까?
    Answers: 일본 | 캐나다 | 한국 (o) | 중국
         
    Your turn!
         
    Context: {context}
""",
        )
    ]
)

question_chain = {"context": format_docs}| question_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: 한국의 수도는 어디입니까?
    Answers: 서울(o) | 부산 | 대구 | 제주

    Question: 비빔밥은 어느 나라 음식입니까?
    Answers: 일본 | 캐나다 | 한국 (o) | 중국
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }}
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }}
                ]
            }},
                        {{
                "question": "한국의 수도는 어디입니까?",
                "answers": [
                        {{
                            "answer": "서울",
                            "correct": true
                        }},
                        {{
                            "answer": "부산",
                            "correct": false
                        }},
                        {{
                            "answer": "대구",
                            "correct": false
                        }},
                        {{
                            "answer": "제주",
                            "correct": false
                        }}
                ]
            }},
            {{
                "question": "비빔밥은 어느 나라 음식입니까?",
                "answers": [
                        {{
                            "answer": "일본",
                            "correct": false
                        }},
                        {{
                            "answer": "캐나다",
                            "correct": false
                        }},
                        {{
                            "answer": "한국",
                            "correct": true
                        }},
                        {{
                            "answer": "중국",
                            "correct": false
                        }}
                ]
            }}
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)

formatting_chain = formatting_prompt | llm

with st.sidebar:
    docs = None
    topic = None
    choice = st.selectbox("무엇을 원하는지 선택하세요 !",(
        "file","Wikipedia Article",
    ),
    )

    ans_choice = st.selectbox("정답을 바로 확인할까요?",("놉!","정답확인 😅"))

    if choice == "file":
        file = st.file_uploader("파일을 업로드하세요.  ** docx, txt , pdf 파일이 가능합니다 **",
                                type=["pdf","txt","docx"])
        
        if file:
            with st.status("Uploading file..."):
                docs = split_file(file)
            
    
    else:
        topic = st.text_input("위키피디아에서 검색할 주제를 입력하세요")
        #lang = st.selectbox("언어를 선택하세요 !",("English","Korean"))
        if topic:
            docs = wiki_search(topic)

            # retriever = WikipediaRetriever(top_k_results=5)
            # with st.status("Searching wikipedia..."):
            #     docs = retriever.get_relevant_documents(topic)
           

if not docs:
    st.markdown(
    """
    Welcome to Quiz GPT 😎 !

    하이하이 ~~ 여러분을 위한 퀴즈 만들기 GPT입니다.
    열심히 외운 영어단어, 시험대비를 위한 문제를 풀어보고 싶은가요?
    퀴즈를 만들고 싶은 자료파일을 업로드해주세요 !
    파일이 없다면, 위키피디아에 원하는 주제로 검색하세요 ! 
    무엇이든 퀴즈로 만들어 드립니다 얄루 ~~

    준비되셨나요?! Let's go ~~~~ ✍️
    """
    )
else:
    response = func_quiz_chain(docs, topic if topic else file.name).additional_kwargs["function_call"]["arguments"]
    response = json.loads(response)
    #response = run_quiz_chain(docs, topic if topic else file.name)
    #st.write(response)
    with st.form("questions_form"):
        for idx, question in enumerate(response["questions"]):
            st.write(f"{idx+1}. "+ question["question"])
            value = st.radio(f"Select an answer", #{idx}", 
                     [answer["answer"] for answer in question["answers"]],
                     index=None, key=f"{idx}_radio")
            
            if ({"answer": value, "correct":True} in question["answers"]):
                st.success("Correct ! 🙆‍♀️")
            elif value is not None:
                if ans_choice == "놉!":
                    st.error("Wrong... 😭")
                else:
                    for answer in question["answers"]:
                        if answer["correct"]:
                            st.error("정답은 " + answer["answer"] +" 입니다 🥹")

        button = st.form_submit_button()
    



