from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from schemas import AgentResponse

load_dotenv()

def main():
    llm = ChatOpenAI(temperature=0)
    tools = [TavilySearch()]
    agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

    response = agent.invoke({"messages": HumanMessage(content="Search for 3 job postings for software engineers in New York on LinkedIn and their details")})
    print(response['messages'][-1].content)

if __name__ == "__main__":
    main()