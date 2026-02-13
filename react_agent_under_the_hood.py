from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI
from langsmith import Client
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool



# client = Client()
# prompt = client.pull_prompt(
#     "hwchase17/react",
#     include_model=True,
# )

# print(prompt)

@tool
def get_text_length(text: str) -> int:
    """
    Returns the length of the text
    """
    return len(text.replace("\n").strip().replace("'", ''))


prompt_text = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:
"""

tools = [get_text_length]
prompt = PromptTemplate.from_template(prompt_text).partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

llm = ChatOpenAI(temperature=0, stop=["\Observation", "Observation"])
agent = {"input": lambda x: x["input"]} | prompt | llm | ReActSingleInputOutputParser()

res = agent.invoke({"input": "What is the text length of 'DOG' in characters?"})
print(res)