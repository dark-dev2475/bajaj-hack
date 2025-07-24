from pydantic import BaseModel
from typing import List

class Justification(BaseModel):
    source: str
    text: str
    relevance: float  # <-- Add this field

class FinalAnswer(BaseModel):
    Decision: str
    PayoutAmount: int
    Confidence: float
    Justifications: List[Justification]
