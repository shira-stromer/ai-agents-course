from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph

load_dotenv()

def main():
    print("Hello ReAct LangGraph!")

AGENT_REASON="agent_reason"
ACT="act"
LAST=-1

@tool
def triple(num: float):
    """Returns the triple of a number."""
    return num * 3


tools = [TavilySearch(max_results=1), triple]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

system_message="You are a helpful assistant that can use tools to answer questions. Use the tools when necessary to find information or perform calculations."

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    response = llm.invoke([{"role": "system", "content": system_message}, *state["messages"]])
    res = {"messages": [response]}
    return res

tool_node = ToolNode(tools=tools)

flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

def should_continue(state: MessagesState) -> str:
    # End if the last message is not a tool response
   # print(len(state["messages"]))
    last = state["messages"][-1]
    # print(last.content, last.tool_calls, len(last.tool_calls) if last.tool_calls else 0)
    res = END if not last.tool_calls else ACT
    return res

flow.add_conditional_edges(AGENT_REASON, should_continue, {END: END, ACT: ACT})
flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="agent_flow.png")


if __name__ == "__main__":
    print("Running agent...")
    res = app.invoke({"messages": [HumanMessage(content="What is the temperature in Tokyo? List it and triple it.")]})
    print(res["messages"][-1].content)