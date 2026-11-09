from pathlib import Path

from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.tools import get_tools
from app.config import get_settings

PROMPTS_DIR = Path(__file__).parent / "prompts"


def get_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.3,
        reasoning_effort=settings.groq_reasoning_effort,
        max_retries=2,
    )


def load_system_prompt() -> str:
    return (PROMPTS_DIR / "system.md").read_text(encoding="utf-8")


def build_graph(checkpointer=None):
    tools = get_tools()
    llm_with_tools = get_llm().bind_tools(tools)
    system_message = SystemMessage(content=load_system_prompt())

    def assistant(state: MessagesState) -> dict:
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile(checkpointer=checkpointer)