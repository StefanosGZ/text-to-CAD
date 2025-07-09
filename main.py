from typing import List, Optional, TypedDict, Union
from pydantic import BaseModel, ValidationError
from langgraph.graph import StateGraph, END
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------
# 1. SCHEMA (CAD JSON Format)
# -------------------------------------
class GraphState(TypedDict):
    input: str
    raw_json: Optional[str]
    cad_json: Optional[dict]
    error: Optional[str]
    valid: Optional[bool]
    note: Optional[str]

class BaseShape(BaseModel):
    type: str
    dimensions: dict

class Feature(BaseModel):
    type: str
    position: Optional[str] = None
    shape: Optional[str] = None
    dimensions: Optional[dict] = None
    diameter: Optional[Union[int, str]] = None
    location: Optional[str] = None

class CADObject(BaseModel):
    base_shape: BaseShape
    features: List[Feature]


# -------------------------------------
# 2. LLM PARSING NODE
# -------------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def llm_parse_node(state):
    prompt = state["input"]
    system = """You are a CAD parser. Convert natural language CAD prompts into a JSON structure with the following format.

            Use snake_case for all keys. Do not use camelCase.
            ...
            
            Example:
            {
              "base_shape": {
                "type": "square",
                "dimensions": {
                  "side_length": "50mm"
                }
              },
              "features": [
                {
                  "type": "hole",
                  "location": "center",
                  "dimensions": {
                    "diameter": "2mm"
                  }
                }
              ]
            }
            """
    user_prompt = f"""
Prompt: {prompt}

Return only valid JSON.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt}
        ]
    )

    output = response.choices[0].message.content
    return {"raw_json": output.strip()}

# -------------------------------------
# 3. VALIDATION NODE
# -------------------------------------

def validate_node(state):
    try:
        parsed = json.loads(state["raw_json"])
        cad_obj = CADObject(**parsed)
        return {"cad_json": cad_obj.model_dump(), "valid": True}
    except (json.JSONDecodeError, ValidationError) as e:
        return {"error": str(e), "valid": False}

# -------------------------------------
# 4. FALLBACK NODE
# -------------------------------------

def fallback_node(state):
    return {
        "cad_json": None,
        "note": "Fallback used due to parsing/validation failure.",
        "error": state.get("error", "Unknown error")
    }

# -------------------------------------
# 5. GRAPH DEFINITION
# -------------------------------------

builder = StateGraph(GraphState)

builder.add_node("parse_prompt", llm_parse_node)
builder.add_node("validate", validate_node)
builder.add_node("fallback", fallback_node)

builder.set_entry_point("parse_prompt")
builder.add_edge("parse_prompt", "validate")

def check_validity(state):
    return "fallback" if state.get("valid") is False else "end"

builder.add_conditional_edges("validate", check_validity, {
    "fallback": "fallback",
    "end": END
})

graph = builder.compile()

# -------------------------------------
# 6. RUN THE AGENT
# -------------------------------------

if __name__ == "__main__":
    user_input = input("Enter a CAD prompt: ")

    initial_state = {"input": user_input}
    result = graph.invoke(initial_state)

    print("\n--- PARSING RESULT ---")
    print(json.dumps(result, indent=2))
