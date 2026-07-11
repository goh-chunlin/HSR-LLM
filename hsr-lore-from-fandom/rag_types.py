from typing import Literal, TypedDict


class LoreChunk(TypedDict):
    title: str
    text: str


class RetrievedChunk(TypedDict):
    title: str
    text: str
    score: float


IntentType = Literal[
    "entity_lookup",
    "timeline_query",
    "relation_query",
    "other",
]


class IntentClassification(TypedDict):
    label: IntentType
    confidence: float
