
# ml_service/schema.py
"""
Defines the Pydantic data model for a structured user query.
"""
from typing import Optional
from pydantic import BaseModel, Field

class PolicyQuery(BaseModel):
    """
    A structured representation of a user's insurance policy query.
    All fields are optional, as they may not be present in every query.
    """
    policy_type: Optional[str] = Field(
        None, 
        description="The category of the policy, e.g., 'Health', 'Personal Accident'."
    )
    claimant_age: Optional[int] = Field(
        None, 
        description="The age of the claimant."
    )
    claimant_gender: Optional[str] = Field(
        None, 
        description="The gender of the claimant, e.g., 'M' or 'F'."
    )
    procedure_or_claim: Optional[str] = Field(
        None, 
        description="The medical procedure or type of claim."
    )
    location: Optional[str] = Field(
        None, 
        description="The city or location where the event occurred."
    )
    policy_duration_months: Optional[int] = Field(
        None, 
        description="The age of the policy in months."
    )
    event_details: Optional[str] = Field(
        None, 
        description="A free-text description of the event or accident."
    )