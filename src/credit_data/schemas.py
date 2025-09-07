from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class BorrowerSchema(BaseModel):
    """Schema describing a borrower master record."""

    borrower_id: int = Field(..., description="Unique borrower identifier")
    state: str = Field(..., min_length=2, max_length=2)
    zip3: Optional[str] = Field(None, description="First 3 digits of ZIP code")
    income_annual: float = Field(..., ge=0)
    employment_tenure_months: int = Field(..., ge=0)
    industry: Optional[str] = None
    education: Optional[str] = None
    household_size: Optional[int] = None
    fico_baseline: int = Field(..., ge=300, le=850)
    credit_utilization_baseline: float = Field(..., ge=0.0, le=1.0)
    prior_delinquencies: int = Field(..., ge=0)
    bank_tenure_months: int = Field(..., ge=0)
    segment: Literal[
        "mass", "affluent", "small_business", "private", "student"
    ] = "mass"


class LoanSchema(BaseModel):
    """Schema describing a loan account at origination."""

    loan_id: int
    borrower_id: int
    product: Literal[
        "card", "auto", "mortgage", "heloc", "personal"
    ]
    origination_dt: date
    maturity_months: int = Field(..., ge=1)
    interest_rate: float = Field(..., ge=0.0)
    orig_balance: float = Field(..., ge=0.0)
    secured_flag: bool
    ltv_at_orig: Optional[float] = Field(None, ge=0.0)
    risk_grade: Optional[str] = None
    underwriting_dti: Optional[float] = Field(None, ge=0.0)
    underwriting_fico: Optional[int] = Field(None, ge=300, le=850)
    channel: Optional[str] = None
    state: Optional[str] = None
    vintage: Optional[str] = None
    credit_limit: Optional[float] = Field(None, ge=0.0)


class LoanMonthlySchema(BaseModel):
    """Schema describing monthly loan performance panel."""

    asof_month: date
    loan_id: int
    borrower_id: int
    product: str
    balance_ead: float = Field(..., ge=0.0)
    scheduled_principal: float = Field(..., ge=0.0)
    current_principal: float = Field(..., ge=0.0)
    current_interest: float = Field(..., ge=0.0)
    utilization: Optional[float] = Field(None, ge=0.0, le=1.0)
    prepay_flag: bool
    days_past_due: int = Field(..., ge=0)
    roll_rate_bucket: Literal["C", "30", "60", "90+", "CO"]
    default_flag: bool
    chargeoff_flag: bool
    recovery_amt: float = Field(..., ge=0.0)
    recovery_lag_m: Optional[int] = Field(None, ge=0)
    cure_flag: bool
    loss_given_default: Optional[float] = Field(None, ge=0.0, le=1.0)
    effective_rate: Optional[float] = Field(None, ge=0.0)
    forbearance_flag: bool = False


class MacroMonthlySchema(BaseModel):
    """Schema for macroeconomic monthly series used by the model."""

    asof_month: date
    unemployment: float
    cpi_yoy: float
    gdp_growth_qoq_ann: float
    fed_funds: float
    treasury_10y: float
    credit_spread_bbb: float
    hpi_yoy: float
