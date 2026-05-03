"""Agent state definition for the ReAct loop."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """LangGraph state for the manual ReAct agent.

    Nodes return partial dicts — only the fields they change.
    ``messages`` uses the ``add_messages`` reducer (append semantics).
    All other fields are plain overwrite (no reducer).
    """

    messages: Annotated[list[BaseMessage], add_messages]
    iteration: int
    max_iterations: int
    last_action: str
    last_action_input: str
    last_observation: str
    final_answer: str
    error: str
