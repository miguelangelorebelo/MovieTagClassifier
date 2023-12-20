from pydantic import BaseModel


class Request(BaseModel):
    question: str = ""
