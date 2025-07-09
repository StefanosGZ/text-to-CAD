from typing import List, Optional, TypedDict, Dict, Type
from pydantic import BaseModel, model_validator, ValidationError
from langgraph.graph import StateGraph, END
from openai import OpenAI
import json
import os
from dotenv import load_dotenv
from shape_schemas import *

load_dotenv()

shape_schemas: Dict[str, Type[BaseModel]] = {
    "rectangle": RectangleDimensions,
    "circle": CircleDimensions,
    "triangle": TriangleDimensions,
    "polygon": PolygonDimensions,
    "ellipse": EllipseDimensions,
    "slot": SlotDimensions,
    "arc": ArcDimensions,
    "box": BoxDimensions,
    "cylinder": CylinderDimensions,
    "cone": ConeDimensions,
    "sphere": SphereDimensions,
    "torus": TorusDimensions,
    "pyramid": PyramidDimensions,
}

# ------------------------------
# 2. MAIN SCHEMAS
# ------------------------------

class BaseShape(BaseModel):
    type: str
    dimensions: dict

    @model_validator(mode="after")
    def validate_shape(cls, values):
        shape_type = values.type
        dims = values.dimensions

        # Normalize "extrusion" -> "thickness"
        if "extrusion" in dims and "thickness" not in dims:
            dims["thickness"] = dims.pop("extrusion")

        if shape_type not in shape_schemas:
            raise ValueError(f"Unsupported shape type: {shape_type}")

        shape_schemas[shape_type](**dims)  # Validate against shape schema
        return values


class Feature(BaseModel):
    type: str
    location: Optional[str] = None
    dimensions: Optional[dict] = None
    shape: Optional[str] = None
    position: Optional[str] = None
    diameter: Optional[str] = None

class CADObject(BaseModel):
    base_shape: BaseShape
    features: List[Feature]

class GraphState(TypedDict):
    input: str
    raw_json: Optional[str]
    cad_json: Optional[dict]
    error: Optional[str]
    valid: Optional[bool]
    note: Optional[str]

# ------------------------------
# 3. LLM PARSING NODE
# ------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def llm_parse_node(state: GraphState):
    prompt = state["input"]

    supported_shapes = ", ".join(shape_schemas.keys())

    system = f"""You are a CAD parser. Convert natural language CAD prompts into a valid JSON object describing the base shape and its features.

    - Always use snake_case for all keys.
    - The output JSON must contain:
      - A `base_shape` field with:
        - `type`: the name of the shape (e.g., "box", "circle", "pyramid")
        - `dimensions`: a dictionary of all shape-specific dimensions
      - A `features` list (even if empty)
    - Nest *all* shape-related parameters inside `dimensions` — even if they look top-level.
    - Do not flatten fields like width, height, radius, etc. They **must** go inside the `dimensions` dictionary.
    - You may normalize radius into diameter when applicable (e.g., for circles).
    - Support common synonyms like "base side" → "side_1", "extrude" → "thickness", etc.
    - Normalize all units to mm (e.g., 1cm → 10mm, 1 inch → 25.4mm)
    - For pyramids, always structure the base shape like this:
    - `base_shape`: the name of the base (e.g., "square", "triangle")
    - `base_dimensions`: a dictionary matching the base shape's required fields
    - For square base: `side_length` or all four `side_*` values
    - For triangle base: `side_1`, `side_2`, `side_3`
    
    Return only valid JSON.
    
    Example:

    Input: "Create a circle with a radius of 10mm and a 2mm hole at the center"
    Output:
    {{
      "base_shape": {{
        "type": "circle",
        "dimensions": {{
          "radius": "10mm",
          "thickness": "5mm"
        }}
      }},
      "features": [
        {{
          "type": "hole",
          "location": "center",
          "dimensions": {{
            "diameter": "2mm"
          }}
        }}
      ]
    }}
    """

    user_prompt = f"Prompt: {prompt}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt}
        ]
    )

    output = response.choices[0].message.content
    return {"raw_json": output.strip()}

# ------------------------------
# 4. VALIDATION NODE
# ------------------------------

def validate_node(state: GraphState):
    try:
        parsed = json.loads(state["raw_json"])
        cad_obj = CADObject(**parsed)
        return {"cad_json": cad_obj.model_dump(), "valid": True}
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        return {"error": str(e), "valid": False}

# ------------------------------
# 5. FALLBACK NODE
# ------------------------------

def fallback_node(state: GraphState):
    return {
        "cad_json": None,
        "note": "Fallback used due to parsing/validation failure.",
        "error": state.get("error", "Unknown error")
    }

# ------------------------------
# 6. GRAPH DEFINITION
# ------------------------------

builder = StateGraph(GraphState)

builder.add_node("parse_prompt", llm_parse_node)
builder.add_node("validate", validate_node)
builder.add_node("fallback", fallback_node)

builder.set_entry_point("parse_prompt")
builder.add_edge("parse_prompt", "validate")

def check_validity(state: GraphState):
    return "fallback" if state.get("valid") is False else "end"

builder.add_conditional_edges("validate", check_validity, {
    "fallback": "fallback",
    "end": END
})

graph = builder.compile()

# ------------------------------
# 7. MAIN EXECUTION
# ------------------------------

if __name__ == "__main__":
    user_input = input("Enter a CAD prompt: ")
    initial_state = {"input": user_input}
    result = graph.invoke(initial_state)

    print("\n--- PARSING RESULT ---")
    print(json.dumps(result, indent=2))
