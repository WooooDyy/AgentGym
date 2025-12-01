"""
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel


class StepRequestBody(BaseModel):
    """Request body for /step endpoint."""

    id: int
    action: str


class ResetRequestBody(BaseModel):
    """Request body for /reset endpoint."""

    id: int
    data_idx: int


class CloseRequestBody(BaseModel):
    """Request body for /close endpoint."""

    id: int
