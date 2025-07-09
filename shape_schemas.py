from typing import Optional
from pydantic import BaseModel, model_validator

# ------------------------------
# 1. SHAPE-SPECIFIC SCHEMAS
# ------------------------------

class RectangleDimensions(BaseModel):
    width: str
    height: str
    thickness: str

class CircleDimensions(BaseModel):
    diameter: Optional[str] = None
    radius: Optional[str] = None
    thickness: str

    @model_validator(mode="after")
    def check_radius_or_diameter(self):
        if not self.diameter and not self.radius:
            raise ValueError("Either 'radius' or 'diameter' must be provided.")

        if not self.diameter and self.radius:
            try:
                r = float(self.radius.replace("mm", "").strip())
                self.diameter = f"{2 * r}mm"
            except Exception:
                raise ValueError("Could not parse radius value")

        return self


class TriangleDimensions(BaseModel):
    side_1: str
    side_2: str
    side_3: str
    thickness: str

class PolygonDimensions(BaseModel):
    number_of_sides: int
    side_length: str
    thickness: str

class EllipseDimensions(BaseModel):
    major_axis: str
    minor_axis: str
    thickness: str

class SlotDimensions(BaseModel):
    length: str
    width: str
    corner_radius: str
    thickness: str

class ArcDimensions(BaseModel):
    radius: str
    angle: str
    thickness: str

class BoxDimensions(BaseModel):
    width: str
    height: str
    depth: str

class CylinderDimensions(BaseModel):
    diameter: str
    height: str

class ConeDimensions(BaseModel):
    base_diameter: str
    top_diameter: str
    height: str

class SphereDimensions(BaseModel):
    diameter: str

class TorusDimensions(BaseModel):
    major_radius: str
    minor_radius: str

class PyramidDimensions(BaseModel):
    base_shape: str               # "square", "triangle", etc.
    base_dimensions: dict         # uses a nested validated shape if needed
    height: str
