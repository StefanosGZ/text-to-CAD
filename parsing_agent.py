from typing import List, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# ---------------------------------------------------------------------------
# ❶ Environment & Configuration
# ---------------------------------------------------------------------------
load_dotenv()  # Loads OPENAI_API_KEY, etc.

# ---------------------------------------------------------------------------
# ❷ State Definition
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: List[HumanMessage]

# ---------------------------------------------------------------------------
# ❹ System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = ( """
    You are a CAD parsing agent. Convert the user's natural-language CAD description into a well-formed JSON object
    with normalized dimensions and explicit feature placements. Use the following structure:

    {
      "raw_prompt": <string>,
      "requested_object": <string>,              # e.g. "half_circle", "gearwheel", "bracket"
      "dimensions": { <key>: <value_mm> },       # All dimension values in millimetres
      "features": [                              # Optional list of added features
        {
          "type": <string>,                      # e.g. "hole"
          "dimension": <value_mm>,
          "placement": {
            "x": <mm>, "y": <mm>, "z": <mm>
          }
        }
      ],
      "material": <string|null>,
      "metadata": {
        "units_detected": <string>,              # e.g. "cm", "mm", "cm and mm"
        "shape_category": "2D" | "3D" | "mechanical_part",
        "operation_intent": <string|null>        # e.g. "extrude", "revolve", "cut"
      }
    }

    Parsing rules:
    • If draw or sketch is mentioned assume that it is always a 2D object.
    • Identify the **main object** the user wants (e.g. "half circle", "gearwheel") and assign it to "requested_object".
    • If the user describes an operation (e.g. "extrude a half circle"), set "operation_intent" in metadata to "extrude",
      but keep "requested_object" as "half_circle". Do not use operations like "extrusion" as the object name.
    • All units must be normalized to millimetres. If the prompt includes values in cm or inches, convert them using the math tools provided.
    • All features must include explicit numeric coordinates for placement. If the user says "in the center", compute the position numerically.
      Assume the origin (0, 0, 0) is the lower-left-front corner of the object unless otherwise specified.
    • If placement is not mentioned, default to x=0, y=0, z=0.
    • If the user says "thickness of 1 cm" or "z dimension is 10", extract that value and normalize it to millimetres.
    • If thickness is already provided in any form, do NOT ask for it again.
    • If the object is 3D and no thickness (z-dimension) is provided, do not return JSON. Instead, ask:
      "What thickness (in mm) should I use for this 3-D object?"
    • If the user responds with a thickness (e.g. "5mm"), treat it as a continuation of the previous prompt and return the full JSON with
      "thickness" added to the "dimensions".
    • Output **only** the resulting JSON. Do not include explanations, extra comments, or markdown.
    """

)

# ---------------------------------------------------------------------------
# ❺ LLM Binding
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o")

# ---------------------------------------------------------------------------
# ❻ Processing Node
# ---------------------------------------------------------------------------
def process(state: AgentState) -> AgentState:
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(msgs)
    print("\n⮕ Agent:")
    print(response.content)  # Either JSON or the follow-up thickness question
    return state

# ---------------------------------------------------------------------------
# ❼ LangGraph Construction
# ---------------------------------------------------------------------------

graph = StateGraph(AgentState)
graph.add_node("process", process)

graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

# ---------------------------------------------------------------------------
# ❽ REPL with Message History Tracking
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🔧 CAD Prompt-to-JSON Agent (type 'exit' to quit)\n")

    # 1. Initialize full message history
    history: List[HumanMessage] = []

    # 2. Interactive loop
    user_input = input("Enter CAD prompt: ")
    while user_input.lower() != "exit":
        history.append(HumanMessage(content=user_input))  # Add to history
        agent.invoke({"messages": history})
        print()
        user_input = input("Enter CAD prompt: ")

