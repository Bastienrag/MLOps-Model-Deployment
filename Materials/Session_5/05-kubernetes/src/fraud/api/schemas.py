"""API contract: Pydantic rejects any out-of-domain input with a 422, before it reaches the model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

# V1..V28 are anonymised PCA components. Observed range in the Kaggle data is about
# [-115, 125]: anything far outside is a client bug, not a transaction.
_PCA_FIELDS = {f"v{i}": (float, Field(ge=-200, le=200)) for i in range(1, 29)}


class _TransactionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(min_length=1, max_length=64)
    customer_id: str | None = Field(
        default=None, max_length=64, description="Personal data: never logged in clear, never used by the model"
    )
    amount: float = Field(ge=0, le=50_000, description="Amount in euros")


Transaction = create_model("Transaction", __base__=_TransactionBase, **_PCA_FIELDS)


class Prediction(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    model_version: str


class Feedback(BaseModel):
    """True label received later (customer dispute, investigation): the "outcome" layer."""

    transaction_id: str
    predicted_fraud: bool
    actual_fraud: bool


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    decision_threshold: float
