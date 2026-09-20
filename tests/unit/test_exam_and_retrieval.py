"""Unit tests for multi-chapter query extraction and exam parameter parsing."""

from math_teacher.retrieval.service import analyze_query
from math_teacher.agents.exam import _extract_exam_params


def test_analyze_query_multi_chapter():
    ctx = analyze_query("conduct an exam for class 9 th use chapter 2 and 7")
    assert ctx.class_level == 9
    assert "Polynomials" in ctx.chapters
    assert "Triangles" in ctx.chapters
    assert ctx.chapter == "Polynomials, Triangles"


def test_extract_exam_params_ignores_assistant_history():
    query = "conduct an exam for class 9 th use chapter 2 and 7"
    chat_history = [
        {
            "role": "assistant",
            "content": "1. **Total Marks:** 20 Marks\n2. **Number of Questions:** 3 Questions\n3. **Difficulty:** Medium",
        }
    ]
    params = _extract_exam_params(query, chat_history)
    # Since query doesn't specify marks or questions, params should be None!
    assert params["total_marks"] is None
    assert params["num_questions"] is None
    assert params["difficulty"] is None
    assert params["is_default"] is False


def test_extract_exam_params_explicit_user_input():
    query = "conduct an exam with 50 marks, 5 questions, hard difficulty"
    params = _extract_exam_params(query, None)
    assert params["total_marks"] == "50 Marks"
    assert params["num_questions"] == "5 Questions"
    assert params["difficulty"] == "Hard (Board Standard)"


def test_analyze_query_typo_chapter():
    ctx = analyze_query("i wnat to know what are the topics there in chaper 9")
    assert ctx.class_level == 9
    assert ctx.chapter in ("Areas of Parallelograms and Triangles", "Circles")


def test_analyze_query_chapter_from_chat_history():
    """When user replies with exam options like '70 marks, 10 questions, hard', chapter context must be preserved from chat history."""
    query = "70 marks, 10 questions, hard"
    chat_history = [
        {"role": "user", "content": "create a question paper for chapter 9 from class 9"},
        {"role": "assistant", "content": "Exam Controller — Mandatory Exam Preparation Settings..."}
    ]
    ctx = analyze_query(query, chat_history=chat_history)
    assert ctx.class_level == 9
    assert ctx.chapter == "Areas of Parallelograms and Triangles"


