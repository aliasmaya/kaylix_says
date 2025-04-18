import os

from langchain_ollama import ChatOllama
from it_state import State, graph_builder
from it_tool import tools, tool_node
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOllama(model=os.getenv("MODEL"), base_url=os.getenv("BASE_URL"))
llm_tools = llm.bind_tools(tools)

memory = MemorySaver()

def chatbot(state: State):
    return {"messages": [llm_tools.invoke(state["messages"])]}

def route_tools(state: State) -> str:
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "kb" and "output" in tool_call:
                kb_output = tool_call["output"]
                if isinstance(kb_output, dict) and kb_output.get("is_empty", True):
                    messages.append(HumanMessage(
                        content=f"The knowledge base had no relevant results for '{kb_output['query']}'. "
                                "Please call the 'tavily' tool."
                    ))
                    return "tools"
        return "tools"

    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        return END

    return "chatbot"

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {
        "tools": "tools",
        "chatbot": "chatbot",
        END: END
    }
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=memory)