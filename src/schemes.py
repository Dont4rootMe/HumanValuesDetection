from pydantic import BaseModel
from typing import List


class MarkupResponse(BaseModel):
    class _Token(BaseModel):
        class _ValuePair(BaseModel):
            label: str
            mode: str
            score: float
            
        start: int
        end: int
        label: List[_ValuePair]

    tokens: List[_Token]
