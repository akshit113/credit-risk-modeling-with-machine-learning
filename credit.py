from pydantic import BaseModel


class Credit(BaseModel):
    revolving_utilization: float
    age: float
    n_30_59_days_past_due: float
    debt_ratio: float
    monthly_income: float
    n_open_credit_lines: float
    n_90_days_late: float
    n_real_estate_loans: float
    n_60_89_past_due: float
    n_dependents: float
