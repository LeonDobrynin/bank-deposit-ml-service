from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class ClientFeatures(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    age: int = Field(ge=18, le=100)
    job: Literal['admin.', 'blue-collar', 'entrepreneur', 'housemaid',
                 'management', 'retired', 'self-employed', 'services',
                 'student', 'technician', 'unemployed', 'unknown']
    marital: Literal['divorced', 'married', 'single', 'unknown']
    education: Literal['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                       'illiterate', 'professional.course', 'university.degree', 'unknown']
    default: Literal['no', 'yes', 'unknown']
    housing: Literal['no', 'yes', 'unknown']
    loan: Literal['no', 'yes', 'unknown']
    pdays: int = Field(ge=0, le=999, description='999 means no previous contact')
    previous: int = Field(ge=0, le=100)
    poutcome: Literal['failure', 'nonexistent', 'success']


class Prediction(BaseModel):
    deposit: Literal[0, 1]
    probability: float = Field(ge=0, le=1)
    threshold: float
    model_version: str
