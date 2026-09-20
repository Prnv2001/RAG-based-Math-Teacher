"""Hybrid retrieval service — vector + keyword → RRF → rerank."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from math_teacher.config.settings import settings
from math_teacher.domain.models import QueryContext, RetrievedChunk
from math_teacher.embeddings.base import Embedder
from math_teacher.storage.models import Document, DocumentChunk


# ---------------------------------------------------------------------------
# Vector Retrieval
# ---------------------------------------------------------------------------


async def vector_search(
    query_embedding: list[float],
    session: AsyncSession,
    class_level: int | None = None,
    chapter: str | None = None,
    top_k: int = 20,
) -> list[tuple[DocumentChunk, float]]:
    """Cosine-similarity search using pgvector HNSW index.

    Applies metadata filters (class_level, chapter) at the DB level.
    Returns list of (chunk, score) ordered by similarity descending.
    """
    # Build embedding literal for pgvector
    vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    filters = ["dc.text IS NOT NULL"]
    params: dict[str, Any] = {"vec": vec_str, "top_k": top_k}

    if class_level is not None:
        filters.append("dc.class_level = :class_level")
        params["class_level"] = class_level

    if chapter:
        ch_names = [c.strip() for c in chapter.split(",") if c.strip()]
        if len(ch_names) > 1:
            ch_or = []
            for idx, ch_name in enumerate(ch_names):
                pname = f"ch_vec_{idx}"
                ch_or.append(f"dc.chapter ILIKE :{pname}")
                params[pname] = f"%{ch_name}%"
            filters.append(f"({' OR '.join(ch_or)})")
        else:
            filters.append("dc.chapter ILIKE :chapter")
            params["chapter"] = f"%{chapter}%"

    where_clause = " AND ".join(filters)

    sql = text(f"""
        SELECT
            dc.id,
            dc.document_id,
            dc.class_level,
            dc.chapter,
            dc.topic,
            dc.concept,
            dc.content_type,
            dc.page_number,
            dc.chunk_index,
            dc.text,
            1 - (dc.embedding <=> CAST(:vec AS vector)) AS score,
            d.title AS document_title
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        WHERE {where_clause}
        ORDER BY dc.embedding <=> CAST(:vec AS vector)
        LIMIT :top_k
    """)

    result = await session.execute(sql, params)
    rows = result.fetchall()

    chunks = []
    for row in rows:
        chunk = RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            document_title=row.document_title,
            class_level=row.class_level,
            chapter=row.chapter,
            topic=row.topic,
            concept=row.concept,
            content_type=row.content_type,
            page_number=row.page_number,
            text=row.text,
            score=float(row.score),
        )
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Keyword Retrieval (PostgreSQL Full-Text Search)
# ---------------------------------------------------------------------------


async def keyword_search(
    query: str,
    session: AsyncSession,
    class_level: int | None = None,
    chapter: str | None = None,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """PostgreSQL full-text search over document_chunks.text.

    Uses ts_rank for scoring.
    """
    filters = ["to_tsvector('english', dc.text) @@ plainto_tsquery('english', :query)"]
    params: dict[str, Any] = {"query": query, "top_k": top_k}

    if class_level is not None:
        filters.append("dc.class_level = :class_level")
        params["class_level"] = class_level

    if chapter:
        ch_names = [c.strip() for c in chapter.split(",") if c.strip()]
        if len(ch_names) > 1:
            ch_or = []
            for idx, ch_name in enumerate(ch_names):
                pname = f"ch_kw_{idx}"
                ch_or.append(f"dc.chapter ILIKE :{pname}")
                params[pname] = f"%{ch_name}%"
            filters.append(f"({' OR '.join(ch_or)})")
        else:
            filters.append("dc.chapter ILIKE :chapter")
            params["chapter"] = f"%{chapter}%"

    where_clause = " AND ".join(filters)

    sql = text(f"""
        SELECT
            dc.id,
            dc.document_id,
            dc.class_level,
            dc.chapter,
            dc.topic,
            dc.concept,
            dc.content_type,
            dc.page_number,
            dc.text,
            ts_rank(to_tsvector('english', dc.text), plainto_tsquery('english', :query)) AS score,
            d.title AS document_title
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        WHERE {where_clause}
        ORDER BY score DESC
        LIMIT :top_k
    """)

    result = await session.execute(sql, params)
    rows = result.fetchall()

    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            document_title=row.document_title,
            class_level=row.class_level,
            chapter=row.chapter,
            topic=row.topic,
            concept=row.concept,
            content_type=row.content_type,
            page_number=row.page_number,
            text=row.text,
            score=float(row.score),
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    *result_lists: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Merge multiple ranked retrieval lists using RRF.

    score += 1 / (k + rank)   for each list
    Returns merged list sorted by fused score descending.
    """
    scores: dict[uuid.UUID, float] = {}
    chunks: dict[uuid.UUID, RetrievedChunk] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list, start=1):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunks[cid] = chunk

    merged = sorted(chunks.values(), key=lambda c: scores[c.chunk_id], reverse=True)
    # Update scores to fused scores
    for chunk in merged:
        chunk = chunk.model_copy(update={"score": scores[chunk.chunk_id]})

    return merged


# ---------------------------------------------------------------------------
# Simple Reranker (relevance score using substring / keyword overlap)
# ---------------------------------------------------------------------------


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Simple keyword-overlap reranker.

    Boosts chunks that contain query terms as a post-processing step.
    In production this would be replaced by a cross-encoder model.
    """
    top_k = top_k or settings.rerank_top_k
    query_tokens = set(query.lower().split())

    def boost(chunk: RetrievedChunk) -> float:
        text_lower = chunk.text.lower()
        overlap = sum(1 for t in query_tokens if t in text_lower)
        return chunk.score + 0.05 * overlap

    reranked = sorted(chunks, key=boost, reverse=True)
    return reranked[:top_k]


# ---------------------------------------------------------------------------
# Query Analyzer — regex/keyword extraction (V2: no LLM call)
# ---------------------------------------------------------------------------

# CBSE Class 9 chapter names (lowercase for matching)
_CLASS_9_CHAPTERS: dict[str, list[str]] = {
    "Number Systems": ["number system", "real number", "irrational", "rational", "natural number"],
    "Polynomials": ["polynomial", "degree", "zero of polynomial", "remainder theorem", "factor theorem"],
    "Coordinate Geometry": ["coordinate", "cartesian", "abscissa", "ordinate", "x-axis", "y-axis"],
    "Linear Equations in Two Variables": ["linear equation", "two variables", "2 variables"],
    "Introduction to Euclid's Geometry": ["euclid", "axiom", "postulate"],
    "Lines and Angles": ["line", "angle", "transversal", "parallel lines", "vertically opposite"],
    "Triangles": ["triangle", "congruence", "congruent", "sss", "sas", "asa", "aas", "rhs", "cpct"],
    "Quadrilaterals": ["quadrilateral", "parallelogram", "rhombus", "rectangle", "trapezium"],
    "Areas of Parallelograms and Triangles": ["areas of parallelograms and triangles", "parallelogram area", "same base", "same parallels", "area of triangle"],
    "Circles": ["circle", "chord", "arc", "sector", "radius", "diameter", "cyclic", "subtended"],
    "Heron's Formula": ["heron", "area of triangle", "semi-perimeter"],
    "Surface Areas and Volumes": ["surface area", "volume", "cylinder", "cone", "sphere", "hemisphere", "cuboid", "cube"],
    "Statistics": ["mean", "median", "mode", "frequency", "histogram", "bar graph", "ogive"],
    "Probability": ["probability", "random", "experiment", "outcome", "event", "dice", "coin"],
}

# CBSE Class 10 chapter names
_CLASS_10_CHAPTERS: dict[str, list[str]] = {
    "Real Numbers": ["real number", "euclid division", "fundamental theorem of arithmetic", "hcf", "lcm", "prime factorisation"],
    "Polynomials": ["polynomial", "zeroes", "zeros", "quadratic polynomial", "cubic polynomial", "coefficient"],
    "Pair of Linear Equations in Two Variables": ["pair of linear", "simultaneous", "substitution method", "elimination method", "cross multiplication"],
    "Quadratic Equations": ["quadratic equation", "discriminant", "roots", "quadratic formula", "completing the square"],
    "Arithmetic Progressions": ["arithmetic progression", "ap ", "common difference", "nth term", "sum of n terms"],
    "Triangles": ["similar triangle", "similarity", "bpt", "basic proportionality", "pythagoras", "pythagorean"],
    "Coordinate Geometry": ["distance formula", "section formula", "midpoint", "area of triangle coordinate"],
    "Introduction to Trigonometry": ["trigonometry", "sin", "cos", "tan", "cosec", "sec", "cot", "trigonometric ratio"],
    "Some Applications of Trigonometry": ["height and distance", "angle of elevation", "angle of depression"],
    "Circles": ["tangent", "secant", "number of tangent"],
    "Areas Related to Circles": ["area of sector", "area of segment", "length of arc"],
    "Surface Areas and Volumes": ["frustum", "combination of solids"],
    "Statistics": ["mean", "median", "mode", "cumulative frequency", "less than ogive", "more than ogive"],
    "Probability": ["probability", "classical probability", "complementary event"],
}

# Intent keywords
_INTENT_KEYWORDS: dict[str, list[str]] = {
    "START_EXAM": ["exam", "test paper", "question paper", "sample paper", "conduct exam",
                  "create an exam", "create a exam", "take an exam", "generate exam"],
    "SOLVE_PROBLEM": ["solve", "find", "calculate", "compute", "evaluate", "simplify",
                       "factorise", "factorize", "determine", "prove", "show that",
                       "find the value", "find x", "find y"],
    "EXPLAIN_CONCEPT": ["explain", "what is", "what are", "define", "describe",
                        "meaning of", "concept of", "tell me about", "how does",
                        "why is", "why do", "why are", "difference between"],
    "GIVE_HINT": ["hint", "clue", "help me start", "how to start", "where to begin",
                  "give me a hint", "nudge"],
    "CHECK_ANSWER": ["check", "verify", "is this correct", "is this right",
                     "am i right", "is my answer", "did i get", "correct answer"],
    "REVISE_TOPIC": ["list all the chapter", "list all teh chapters", "list all chapters", "list chapters", "list teh chapters", "show chapters",
                    "all chapters", "all teh chapters", "chapter list", "all the chapter", "syllabus", "table of contents",
                    "topics in chapter", "revise", "roadmap", "overview"],
}

# Difficulty keywords
_DIFFICULTY_KEYWORDS: dict[str, list[str]] = {
    "easy": ["easy", "simple", "basic", "beginner"],
    "medium": ["medium", "moderate", "intermediate"],
    "hard": ["hard", "difficult", "tough", "challenging", "advanced", "complex"],
}


def _match_kw(kw: str, text: str) -> bool:
    """Check if keyword matches text. Short terms (<= 4 chars) require word boundaries."""
    if len(kw) <= 4:
        import re
        return bool(re.search(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE))
    return kw.lower() in text.lower()


_CLASS_9_CHAPTER_BY_NUM: dict[int, str] = {
    1: "Number Systems",
    2: "Polynomials",
    3: "Coordinate Geometry",
    4: "Linear Equations in Two Variables",
    5: "Introduction to Euclid's Geometry",
    6: "Lines and Angles",
    7: "Triangles",
    8: "Quadrilaterals",
    9: "Areas of Parallelograms and Triangles",
    10: "Circles",
    11: "Constructions",
    12: "Heron's Formula",
    13: "Surface Areas and Volumes",
    14: "Statistics",
    15: "Probability",
}

_CLASS_10_CHAPTER_BY_NUM: dict[int, str] = {
    1: "Real Numbers",
    2: "Polynomials",
    3: "Pair of Linear Equations in Two Variables",
    4: "Quadratic Equations",
    5: "Arithmetic Progressions",
    6: "Triangles",
    7: "Coordinate Geometry",
    8: "Introduction to Trigonometry",
    9: "Some Applications of Trigonometry",
    10: "Circles",
    11: "Areas Related to Circles",
    12: "Surface Areas and Volumes",
    13: "Statistics",
    14: "Probability",
}


def analyze_query(
    query: str,
    class_level: int | None = None,
    chapter: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
) -> QueryContext:
    """Extract structured QueryContext from a raw student question.

    V2: Uses regex + keyword matching — no LLM call needed. Resolves context from chat_history.
    """
    from math_teacher.domain.enums import Difficulty, Intent

    q_lower = query.lower()

    # --- Class level ---
    detected_class = class_level
    if detected_class is None:
        import re
        m = re.search(r"\bclass\s*(9|10|ix|x)\b", q_lower)
        if m:
            raw = m.group(1)
            detected_class = 9 if raw in ("9", "ix") else 10
        elif re.search(r"\b(9th|ninth)\b", q_lower):
            detected_class = 9
        elif re.search(r"\b(10th|tenth)\b", q_lower):
            detected_class = 10

    # Fallback to chat_history for class level if follow-up query
    if detected_class is None and chat_history:
        import re
        for msg in reversed(chat_history):
            content = msg.get("content", "").lower()
            m = re.search(r"\bclass\s*(9|10|ix|x)\b", content)
            if m:
                raw = m.group(1)
                detected_class = 9 if raw in ("9", "ix") else 10
                break
            elif re.search(r"\b(9th|ninth)\b", content):
                detected_class = 9
                break
            elif re.search(r"\b(10th|tenth)\b", content):
                detected_class = 10
                break

    # --- Chapter ---
    detected_chapter = chapter
    detected_chapters: list[str] = []
    if detected_chapter is None:
        import re
        cls = detected_class or 9
        mapping = _CLASS_10_CHAPTER_BY_NUM if cls == 10 else _CLASS_9_CHAPTER_BY_NUM

        # Check for multi-chapter expressions: e.g. "chapter 2 and 7", "chapters 2, 7", "ch 2 & 7"
        multi_ch_m = re.search(
            r"\b(?:ch|chap|chapt|chptr|chpt|chaper|chapre|chapr|chapter|chapters|chapetr|chpater|capter|chppter|unit|units)\s*[-:]?\s*(\d{1,2}(?:\s*(?:,|&|and)\s*\d{1,2})+)\b",
            q_lower,
        )
        if multi_ch_m:
            num_str = multi_ch_m.group(1)
            ch_nums = [int(n) for n in re.findall(r"\d{1,2}", num_str)]
            for cn in ch_nums:
                name = mapping.get(cn)
                if name and name not in detected_chapters:
                    detected_chapters.append(name)
            if detected_chapters:
                detected_chapter = ", ".join(detected_chapters)
        else:
            # Single chapter search (typo-tolerant)
            ch_m = re.search(
                r"\b(?:ch|chap|chapt|chptr|chpt|chaper|chapre|chapr|chapter|chapters|chapetr|chpater|capter|chppter|unit|units)\s*[-:]?\s*(\d{1,2})\b",
                q_lower,
            )
            if not ch_m:
                ch_m = re.search(r"\bch\w*\s*[-:]?\s*(\d{1,2})\b", q_lower)
            if not ch_m:
                ch_m = re.search(r"\b(?:class\s*(?:9|10|ix|x)|9th|10th)\s+.*?(\d{1,2})\b", q_lower)
            if ch_m:
                ch_num = int(ch_m.group(1))
                name = mapping.get(ch_num)

                if name:
                    detected_chapter = name
                    detected_chapters = [name]
                    if detected_class is None:
                        detected_class = cls

    if detected_chapter is None:
        cls = detected_class or 9
        target_chapters = _CLASS_9_CHAPTERS if cls == 9 else _CLASS_10_CHAPTERS
        other_chapters = _CLASS_10_CHAPTERS if cls == 9 else _CLASS_9_CHAPTERS

        for ch_name, keywords in target_chapters.items():
            if any(_match_kw(kw, q_lower) for kw in keywords):
                detected_chapter = ch_name
                detected_chapters = [ch_name]
                if detected_class is None:
                    detected_class = cls
                break
        if detected_chapter is None:
            for ch_name, keywords in other_chapters.items():
                if any(_match_kw(kw, q_lower) for kw in keywords):
                    detected_chapter = ch_name
                    detected_chapters = [ch_name]
                    if detected_class is None:
                        detected_class = 10 if cls == 9 else 9
                    break

    # Fallback to chat_history for chapter if follow-up query (e.g. user providing exam setup settings)
    if detected_chapter is None and chat_history:
        import re
        cls = detected_class or 9
        mapping = _CLASS_10_CHAPTER_BY_NUM if cls == 10 else _CLASS_9_CHAPTER_BY_NUM
        for msg in reversed(chat_history):
            if msg.get("role") != "user":
                continue
            hist_text = msg.get("content", "").lower()

            # 1. Multi-chapter in history
            multi_ch_m = re.search(
                r"\b(?:ch|chap|chapt|chptr|chpt|chaper|chapre|chapr|chapter|chapters|chapetr|chpater|capter|chppter|unit|units)\s*[-:]?\s*(\d{1,2}(?:\s*(?:,|&|and)\s*\d{1,2})+)\b",
                hist_text,
            )
            if multi_ch_m:
                num_str = multi_ch_m.group(1)
                ch_nums = [int(n) for n in re.findall(r"\d{1,2}", num_str)]
                for cn in ch_nums:
                    name = mapping.get(cn)
                    if name and name not in detected_chapters:
                        detected_chapters.append(name)
                if detected_chapters:
                    detected_chapter = ", ".join(detected_chapters)
                    break

            # 2. Single chapter in history
            ch_m = re.search(
                r"\b(?:ch|chap|chapt|chptr|chpt|chaper|chapre|chapr|chapter|chapters|chapetr|chpater|capter|chppter|unit|units)\s*[-:]?\s*(\d{1,2})\b",
                hist_text,
            )
            if not ch_m:
                ch_m = re.search(r"\bch\w*\s*[-:]?\s*(\d{1,2})\b", hist_text)
            if not ch_m:
                ch_m = re.search(r"\b(?:class\s*(?:9|10|ix|x)|9th|10th)\s+.*?(\d{1,2})\b", hist_text)
            if ch_m:
                ch_num = int(ch_m.group(1))
                name = mapping.get(ch_num)
                if name:
                    detected_chapter = name
                    detected_chapters = [name]
                    break

            # 3. Chapter name keywords in history
            target_chapters = _CLASS_9_CHAPTERS if cls == 9 else _CLASS_10_CHAPTERS
            for ch_name, keywords in target_chapters.items():
                if any(_match_kw(kw, hist_text) for kw in keywords if kw not in ("find", "solve")):
                    detected_chapter = ch_name
                    detected_chapters = [ch_name]
                    break
            if detected_chapter:
                break


    # --- Topic / Concept (use chapter keywords as proxy) ---
    detected_topic = None
    all_chapters = {**_CLASS_9_CHAPTERS, **_CLASS_10_CHAPTERS}
    if detected_chapter:
        first_ch = detected_chapters[0] if detected_chapters else detected_chapter.split(",")[0].strip()
        if first_ch in all_chapters:
            for kw in all_chapters[first_ch]:
                if _match_kw(kw, q_lower):
                    detected_topic = kw.title()
                    break

    # --- Intent ---
    detected_intent: Intent | None = None
    for intent_str, keywords in _INTENT_KEYWORDS.items():
        if any(_match_kw(kw, q_lower) for kw in keywords):
            try:
                detected_intent = Intent(intent_str)
            except ValueError:
                pass
            break

    # --- Difficulty ---
    detected_difficulty: Difficulty | None = None
    for diff_str, keywords in _DIFFICULTY_KEYWORDS.items():
        if any(_match_kw(kw, q_lower) for kw in keywords):
            try:
                detected_difficulty = Difficulty(diff_str)
            except ValueError:
                pass
            break

    return QueryContext(
        original_query=query,
        class_level=detected_class,
        subject="Mathematics",
        chapter=detected_chapter,
        chapters=detected_chapters,
        topic=detected_topic,
        concept=None,
        intent=detected_intent,
        difficulty=detected_difficulty,
    )


# ---------------------------------------------------------------------------
# Full Retrieval Service
# ---------------------------------------------------------------------------


async def retrieve(
    query: str,
    query_context: QueryContext,
    embedder: Embedder,
    session: AsyncSession,
    top_k: int | None = None,
    rerank_top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Full hybrid retrieval: vector + keyword → RRF → rerank.

    Args:
        query: The search query string.
        query_context: Extracted context for filters.
        embedder: For generating the query embedding.
        session: Active DB session.
        top_k: Initial retrieval count (default from settings).
        rerank_top_k: Final count after reranking (default from settings).
    Returns:
        Reranked list of RetrievedChunk (max rerank_top_k items).
    """
    top_k = top_k or settings.retrieval_top_k
    rerank_top_k = rerank_top_k or settings.rerank_top_k

    # Meta-Curriculum Check: If asking for chapter listing/syllabus, return full class chapter overview chunk (typo-tolerant)
    q_low = query.lower()
    import re
    is_meta_chapter_query = bool(
        re.search(r"\b(?:list|show|all|give|get|view|tell)\b.*?\b(?:chapt\w*|chptr\w*|chaper\w*|capter\w*|chapter\w*|syllab\w*|toc|topics)\b", q_low)
    ) or any(k in q_low for k in ("table of contents", "all chapters", "all chapter", "chapter list", "full syllabus"))

    if is_meta_chapter_query:
        cls = query_context.class_level or 9
        mapping = _CLASS_10_CHAPTER_BY_NUM if cls == 10 else _CLASS_9_CHAPTER_BY_NUM
        chapters_text = "\n".join(f"Chapter {num}: {name}" for num, name in mapping.items())
        meta_chunk = RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            document_title=f"CBSE Class {cls} Mathematics Official Chapter List",
            class_level=cls,
            chapter="All Chapters",
            topic="Curriculum Syllabus Overview",
            concept=None,
            content_type="summary",
            page_number=1,
            text=f"Official CBSE NCERT Class {cls} Mathematics Table of Contents:\n{chapters_text}",
            score=1.0,
        )
        return [meta_chunk]

    # Embed query
    query_embedding = await embedder.embed(query)

    try:
        # Run vector and keyword search
        vector_results = await vector_search(
            query_embedding=query_embedding,
            session=session,
            class_level=query_context.class_level,
            chapter=query_context.chapter,
            top_k=top_k,
        )

        keyword_results = await keyword_search(
            query=query,
            session=session,
            class_level=query_context.class_level,
            chapter=query_context.chapter,
            top_k=top_k,
        )

        # Merge with RRF
        fused = reciprocal_rank_fusion(vector_results, keyword_results)

        # Fallback: if chapter filter returned 0 results, retry without chapter filter
        if not fused and query_context.chapter:
            vector_results = await vector_search(
                query_embedding=query_embedding,
                session=session,
                class_level=query_context.class_level,
                chapter=None,
                top_k=top_k,
            )
            keyword_results = await keyword_search(
                query=query,
                session=session,
                class_level=query_context.class_level,
                chapter=None,
                top_k=top_k,
            )
            fused = reciprocal_rank_fusion(vector_results, keyword_results)

        # Fallback: if still 0 results (e.g. class_level filter too strict), search whole DB
        if not fused and query_context.class_level:
            vector_results = await vector_search(
                query_embedding=query_embedding,
                session=session,
                class_level=None,
                chapter=None,
                top_k=top_k,
            )
            keyword_results = await keyword_search(
                query=query,
                session=session,
                class_level=None,
                chapter=None,
                top_k=top_k,
            )
            fused = reciprocal_rank_fusion(vector_results, keyword_results)

        # Rerank
        final = rerank(query=query, chunks=fused, top_k=rerank_top_k)
        return final
    except Exception as exc:
        import logging
        logging.getLogger("math_teacher.retrieval").warning(f"Database retrieval unavailable ({exc}). Continuing with LLM knowledge base.")
        return []

