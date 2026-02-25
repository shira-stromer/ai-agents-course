from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilyResearch

load_dotenv()

def main():
    print("Hello ReAct LangGraph!")


@tool
def triple(num: float):
    """Returns the triple of a number."""
    return num * 3


tools = [TavilyResearch(max_results=1), triple]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)



if __name__ == "__main__":
    main()