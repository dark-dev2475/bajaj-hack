from pydantic import BaseModel, Field
from typing import List

class Justification(BaseModel):
    source: str
    text: str
    relevance: float  # <-- Add this field

class FinalAnswer(BaseModel):
    """
    An enhanced structured representation of the final decision from the LLM.
    """
    Decision: str = Field(..., description="The final decision, e.g., 'Covered', 'Not Covered', 'Partial Coverage'.")
    Reasoning: str = Field(..., description="A step-by-step explanation of how the decision was reached.")
    # NextSteps: str = Field(..., description="Actionable next steps for the user.")
    PayoutAmount: int = Field(..., description="The estimated payout amount. Can be 0 if not covered.")
    Confidence: float = Field(..., ge=0.0, le=1.0, description="The confidence score (0-1) of the decision.")
    Justifications: List[Justification] = Field(..., description="An array of clauses supporting the decision.")