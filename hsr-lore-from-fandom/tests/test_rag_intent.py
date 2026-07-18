from rag_intent import (
    build_intent_addendum,
    classify_query_intent,
    retrieval_score_adjustment,
    retrieval_top_k_for_intent,
)


def test_classify_query_intent_entity_lookup() -> None:
    result = classify_query_intent("Who is Dan Heng?")
    assert result["label"] == "entity_lookup"
    assert result["confidence"] >= 0.9


def test_classify_query_intent_timeline_query() -> None:
    result = classify_query_intent("When did Penacony events happen?")
    assert result["label"] == "timeline_query"


def test_classify_query_intent_relation_query_priority() -> None:
    result = classify_query_intent("When did Kafka and Blade relationship begin?")
    assert result["label"] == "relation_query"


def test_retrieval_top_k_for_intent() -> None:
    assert retrieval_top_k_for_intent("timeline_query", 3) == 5
    assert retrieval_top_k_for_intent("relation_query", 4) == 5
    assert retrieval_top_k_for_intent("entity_lookup", 4) == 4


def test_retrieval_score_adjustment_entity_title_match_bonus() -> None:
    score = retrieval_score_adjustment(
        intent="entity_lookup",
        query="who is silver wolf",
        title="silver wolf",
        text="a member of stellaron hunters",
    )
    assert score >= 0.15


def test_retrieval_score_adjustment_timeline_year_bonus() -> None:
    score = retrieval_score_adjustment(
        intent="timeline_query",
        query="what happened in 2024",
        title="history",
        text="in 2024, the event happened",
    )
    assert score >= 0.2


def test_build_intent_addendum_has_mode_label() -> None:
    assert "ENTITY LOOKUP" in build_intent_addendum("entity_lookup")
    assert "TIMELINE" in build_intent_addendum("timeline_query")
    assert "RELATION" in build_intent_addendum("relation_query")
    assert "OTHER" in build_intent_addendum("other")
