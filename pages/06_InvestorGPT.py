import streamlit as st
from typing import Type
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import DuckDuckGoSearchAPIWrapper, GoogleSearchAPIWrapper
import os
import requests
from langchain.schema import SystemMessage


google_api_key = st.text_input(label="GOOGLE_API_KEY")
google_cse_id = st.text_input(label="GOOGLE CSE ID")
alpha_vantage_api_key = st.text_input(label="Alpha Vantage api key")
openai_api_key = st.text_input(label="OPENAI_API_KEY")

os.environ["GOOGLE_CSE_ID"] = google_api_key
os.environ["GOOGLE_API_KEY"] = google_cse_id
os.environ["OPENAI_API_KEY"] = openai_api_key

#Use Alpha Vantage API
# alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
# os.environ["GOOGLE_CSE_ID"] = os.environ.get("GOOGLE_CSE_ID")
# os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")


llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo-1106")

st.set_page_config(
    page_icon="💰",
    page_title="Investor GPT",
)

st.markdown(
    """
    # Investor GPT
    
    ### Welcome to Investor GPT !
    
    #### 이 회사 주식 사? 말아? 살말 고민이라면? \n
    아래 회사 이름을 입력하면 주식도우미 Investor GPT에게
    주식 정보와 함꼐 추천/비추천 제안을 받을 수 있어요 😎
    
    """
)


class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(
        description= "The query you will search for. Example query : \
            Stock market symbol for Google"
    )

class StockMarketSymbolSearchTool(BaseTool):
    name = "StockMarketSymbolSearchTool"
    description = """
    Use this tool to find the stock market symbol for a company.
    It takes a query as an argument.
    """  
    args_schema: Type[StockMarketSymbolSearchToolArgsSchema] = StockMarketSymbolSearchToolArgsSchema 
    def _run(self, query):
        search = GoogleSearchAPIWrapper()
        return search.run(query)

class CompanyOverviewToolArgsSchema(BaseModel):
    symbol: str = Field(
        description= "Stock symbol of the company. Example : TSLA, AAPL"
    )
    
class CompanyOverviewTool(BaseTool):
    name = "CompanyOverviewTool"
    description = """
    Use this to get an overview of the financials of the company.
    You should enter a stock symbol.
    """
    
    args_schema : Type[CompanyOverviewToolArgsSchema] = CompanyOverviewToolArgsSchema
    
    def _run(self, symbol):
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}"
        r = requests.get(url)
        return r.json()
    
class CompanyIncomeStatementTool(BaseTool):
    name = "CompanyIncomeStatement"
    description = """
    Use this to get the income statement of a company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewToolArgsSchema] = CompanyOverviewToolArgsSchema
    
    def _run(self, symbol):
        url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}"
        r = requests.get(url)
        return r.json()["annualReports"]

class CompanyStockPerformanceTool(BaseTool):
    name = "CompanyStockPerformance"
    description = """
    Use this to get the weekly performance of a company stock.
    You should enter a stock symbol.
    """
    
    args_schema: Type[CompanyOverviewToolArgsSchema] = CompanyOverviewToolArgsSchema
    
    def _run(self, symbol):
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}"
        r = requests.get(url)
        response = r.json()
        return list(response["Weekly Time Series"].items())[:200]
        

agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent = AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools = [
        StockMarketSymbolSearchTool(),
        CompanyOverviewTool(),
        CompanyIncomeStatementTool(),
        CompanyStockPerformanceTool(),
    ],
    agent_kwargs=
        {"system_message": SystemMessage(content ="""
        너는 헷지 펀드 매니저야.
        너는 회사를 분석해서 이 주식을 사야할지 말지에 대한 의견과 그 이유를 
        말해줘야해.
        주식의 실적, 회사의 개요 그리고 손익계산서를 고려해줘.
        너의 판단에 확신을 가지고 주식 구매를 추천하거나 반대해줘.
        답변은 한국어로 작성해줘.
        """ )}
)

# prompt = "Give me financial information on Tesla's stock, \
#     considering its financials, income statements and \
#     stock performance help me analyze if it's a potential good investment."

#print(agent.invoke(prompt))


company = st.text_input("관심있는 회사 이름을 입력하세요 !")


if company:
    
    result = agent.invoke(company)
    st.write(result["output"].replace("$","\$"))
    

    