from typing import Any

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langsmith import Client
import time
from schemas import AgentResponse
from langchain_core.runnables import RunnableLambda, RunnableParallel


load_dotenv()

client = Client()
prompt = client.pull_prompt("hwchase17/react")
output_parser = PydanticOutputParser[AgentResponse](pydantic_object=AgentResponse)

def text_to_upper_case_task(input_text: str) -> str:
    time.sleep(3)
    return f"Text {input_text} converted to {input_text.upper()}"

def reverse_text_task(input_text: str) -> str:
    time.sleep(5)
    return f"Text {input_text} converted to {input_text[::-1]}"

def task_to_runnable_lambda(task) -> RunnableLambda:
    return RunnableLambda[Any, Any](task)

def main():
    runnable_task1 = task_to_runnable_lambda(task=text_to_upper_case_task)
    runnable_task2 = task_to_runnable_lambda(task=reverse_text_task)

    parallel_task = RunnableParallel[Any](
        uppcase_text=runnable_task1,
        reverse_text=runnable_task2
    )
    result = parallel_task.invoke("langchain")

    llm = ChatOpenAI(temperature=0)
    structured_llm = llm.with_structured_output(AgentResponse)
    chain = prompt | structured_llm
    response = chain.invoke(input={"messages": HumanMessage(content="Convert the text 'langchain' to uppercase and reverse it")})
    print(response)


if __name__ == "__main__":
    main()