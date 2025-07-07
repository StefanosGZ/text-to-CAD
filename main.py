from typing import List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

# Updated system prompt to require numeric coordinate calculations
SYSTEM_PROMPT = (
    "You are a CAD parsing agent. "
    "Your task is to take the user's natural-language description of a CAD model "
    "and extract all relevant parameters (shapes, dimensions, positions, features, etc.) into "
    "a single, well-formed JSON object. "
    "For every feature’s position, compute and include explicit numeric coordinates (e.g., x, y, z) "
    "rather than using relative terms like 'center'. "
    "Respond with JSON only—no extra explanation."
)

@tool
def add(a: float, b: float) :
    """This is an addition function that adds 2 numbers together"""
    print("Using add")
    return a + b

@tool
def subtract(a: float, b: float):
    """This is an addition function that substracts 2 numbers together"""
    print("Using substract")
    return a - b


@tool
def multiply(a: float, b: float):
    """This is a multiplication function that multiplies two numbers together"""
    print("Using multiply")
    return a * b

@tool
def divide(a: float, b: float):
    """This is a dividing function that divides two numbers from eachother"""
    print("Using divide")
    return a/b

tools = [add, subtract, multiply, divide]

llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def process(state: AgentState) -> AgentState:
    # Always prepend the system prompt before the user's message
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(msgs)
    # Print out the JSON the model returns
    print(f"\nAI: {response.content}")
    return state

# Wire up the simple state graph
tool_node = ToolNode(tools=tools)
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_node("tools", tool_node)
graph.add_edge(START, "process")
graph.add_conditional_edges("process", tools_condition)
graph.add_edge("tools", "process")
graph.add_edge("process", END)
agent = graph.compile()

# Run the REPL
if __name__ == "__main__":
    user_input = input("Enter CAD prompt (or 'exit'): ")
    while user_input.lower() != "exit":
        agent.invoke({"messages": [HumanMessage(content=user_input)]})
        user_input = input("Enter CAD prompt (or 'exit'): ")
