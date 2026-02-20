from langchain_core.callbacks import BaseCallbackHandler
from typing import Any
from langchain_core.outputs import LLMResult

class AgentCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> Any:
        print(f"***Prompt to LLM was:***\n{prompts[0]}")
        print("********************")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        print(f"***Prompt response:***\n{response.generations[0][0].text}")
        print("********************")