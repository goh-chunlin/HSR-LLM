from typing import Literal, NotRequired, TypedDict


class ReferenceMetadata(TypedDict):
    sourceName: str
    sourceUrl: str
    pageId: NotRequired[str]
    revisionId: NotRequired[str]
    retrievedAt: NotRequired[str]
    license: NotRequired[str]


MediaType = Literal["image", "video"]


class MediaMetadata(TypedDict):
    url: str
    type: MediaType
    title: NotRequired[str]
    description: NotRequired[str]
    attributionUrl: NotRequired[str]
    copyrightOrLicense: NotRequired[str]


class LoreChunk(TypedDict):
    title: str
    text: str
    source: NotRequired[str]
    chunk_key: NotRequired[str]
    reference: NotRequired[ReferenceMetadata]
    media: NotRequired[list[MediaMetadata]]


class RetrievedChunk(TypedDict):
    title: str
    text: str
    score: float
    reference: NotRequired[ReferenceMetadata]
    media: NotRequired[list[MediaMetadata]]


IntentType = Literal[
    "entity_lookup",
    "timeline_query",
    "relation_query",
    "other",
]


class IntentClassification(TypedDict):
    label: IntentType
    confidence: float
