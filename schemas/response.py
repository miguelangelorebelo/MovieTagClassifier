from pydantic import BaseModel


class Result(BaseModel):
    response: str
    elapsed_seconds: float
