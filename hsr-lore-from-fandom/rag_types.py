from typing import TypedDict


class LoreChunk(TypedDict):
    title: str
    text: str


class RetrievedChunk(TypedDict):
    title: str
    text: str
    score: float
