from pydantic import BaseModel, Field

class Source(BaseModel):
    """Schema for a source of information used by the agent."""

    url: str = Field(description="The URL of the source.")

class AgentResponse(BaseModel):
    """Schema for the response from the agent."""

    answer: str = Field(description="The agent's answer to the user's query.")
    sources: list[Source] = Field(default_factory=list, description="A list of sources used by the agent.")

