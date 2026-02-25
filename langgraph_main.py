from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilyResearch
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

load_dotenv()

def main():
    print("Hello ReAct LangGraph!")


@tool
def triple(num: float):
    """Returns the triple of a number."""
    return num * 3


tools = [TavilyResearch(max_results=1), triple]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

system_message="You are a helpful assistant that can use tools to answer questions. Use the tools when necessary to find information or perform calculations."

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    response = llm({"role": "system", "content": system_message}, *state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools=tools)

if __name__ == "__main__":
    main()