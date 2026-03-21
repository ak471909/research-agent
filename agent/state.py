from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Single source of truth passed between all nodes in the graph.

    messages: full conversation history. add_messages is a reducer
              that appends new messages rather than overwriting the list.
    sources:  URLs the agent found and used, collected at the end.
    """

    messages: Annotated[list, add_messages]
    sources: list[str]
    iterations: int
    read_page_count: int
