"""Text sanitizer utility.

Strips raw LaTeX commands, backslashes, and TeX tokens from LLM output,
ensuring clean, human-readable math responses for students.
"""

from __future__ import annotations

import re


_SUPERSCRIPTS = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
    "n": "ⁿ", "i": "ⁱ", "x": "ˣ", "a": "ᵃ", "b": "ᵇ"
}

_SUBSCRIPTS = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
    "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
    "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ",
    "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
    "v": "ᵥ", "x": "ₓ"
}

_GREEK = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\theta": "θ", r"\lambda": "λ", r"\mu": "μ",
    r"\pi": "π", r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ",
    r"\phi": "φ", r"\omega": "ω", r"\Delta": "Δ", r"\Sigma": "Σ",
    r"\Omega": "Ω", r"\Pi": "Π"
}

_OPERATORS = [
    (r"\\[lL]ongrightarrow|\\[rR]ightarrow|\\implies", " ⇒ "),
    (r"\\[lL]ongleftarrow|\\[lL]eftarrow", " ⇐ "),
    (r"\\[lL]ongleftrightarrow|\\iff|\\Leftrightarrow", " ⇔ "),
    (r"\\[lL]ongleftrightarrows|\\leftrightarrow|≤ftrightarrow|\bleftrightarrow\b", " ↔ "),
    (r"\\to|\\rightarrow", " → "),
    (r"\\cdot", "·"),
    (r"\\times", "×"),
    (r"\\div", "÷"),
    (r"\\pm", "±"),
    (r"\\mp", "∓"),
    (r"\\leq?\b", "≤"),
    (r"\\geq?\b", "≥"),
    (r"\\neq\b", "≠"),
    (r"\\approx\b", "≈"),
    (r"\\equiv\b", "≡"),
    (r"\\infty\b", "∞"),
    (r"\\degree|\^\s*\\circ|\^\{?\\circ\}?|\^°", "°"),
    (r"\\in\b", "∈"),
    (r"\\notin\b", "∉"),
    (r"\\subset\b", "⊂"),
    (r"\\cap\b", "∩"),
    (r"\\cup\b", "∪"),
]


def clean_math_text(text: str) -> str:
    """Sanitize and convert LaTeX/math notation into 100% human-readable text.

    Converts fractions, superscripts, subscripts, radicals, operators, Greek letters,
    strips dollar signs ($x$ -> x), removes TeX backslashes and formatting environments.

    Args:
        text: Raw response string from LLM or context.

    Returns:
        100% human-readable plain text math string.
    """
    if not text:
        return ""

    # 0. Formatting wrappers & unit tags: {rm cm}, \mathrm{cm}, boxed{...}, malformed {22{7}
    text = re.sub(r"\{?\\?rm\s*([^}]+)\}", r"\1", text)
    text = re.sub(r"\\?boxed\{([^}]+)\}", r"[\1]", text)
    text = re.sub(r"\{(\d+)\{(\d+)\}", r"(\1/\2)", text)
    text = re.sub(r"\{(\d+)/(\d+)\}", r"(\1/\2)", text)


    # 1. Strip display math delimiters & wrappers
    text = re.sub(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", "", text)
    text = re.sub(r"\\left\(|\\right\)", lambda m: "(" if "left" in m.group(0) else ")", text)
    text = re.sub(r"\\left\[|\\right\]", lambda m: "[" if "left" in m.group(0) else "]", text)
    text = re.sub(r"\\left\\\{|\\right\\\}", lambda m: "{" if "left" in m.group(0) else "}", text)
    text = re.sub(r"\\left\||\\right\|", "|", text)
    text = re.sub(r"\\left\.|\\right\.", "", text)
    text = re.sub(r"\\[\[\]\(\)]", "", text)

    # 2. Text/Font formatting wrappers: \text{...}, \mathrm{...}, \mathbf{...}, \operatorname{...}
    text = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\(?:displaystyle|textstyle|scriptstyle)", "", text)

    # 3. Geometry symbols: triangle ABC -> △ABC, angle ABC -> ∠ABC, cong -> ≅, perpendicular -> perpendicular
    text = re.sub(r"\\?triangle\s+([A-Z]{1,3}\b)", r"△\1", text)
    text = re.sub(r"\\?\bangle\b\s+([A-Z]{1,3}\b)", r"∠\1", text)
    text = re.sub(r"\\cong|\bcong\b(?!\s*r?uent|\s*r?uence)", " ≅ ", text)
    text = re.sub(r"\\?perpendicular\b|\\?perp[\s\-]*endicular\b|⊥[\s\-]*endicular\b|perp[\s\-]*endicular\b", "perpendicular", text)
    text = re.sub(r"\\perp\b|\bperp\b(?!\s*endicular)", " ⊥ ", text)
    text = re.sub(r"\\?parallel", " ∥ ", text)

    # 4. Fractions
    def _format_part(expr: str) -> str:
        expr = expr.strip()
        if not expr:
            return expr
        # Single token like x, 1, 3x^4, 2a, b^2
        if re.match(r"^[a-zA-Z0-9^._⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ]+$", expr) or (expr.startswith("(") and expr.endswith(")")):
            return expr
        return f"({expr})"

    def _replace_curly_frac(m: re.Match) -> str:
        sign = m.group(1) or ""
        raw_num = m.group(2).strip()
        raw_den = m.group(3).strip()
        num = _format_part(raw_num)
        den = _format_part(raw_den)
        if num.startswith("(") or den.startswith("("):
            return f"{sign}{num}/{den}"
        return f"{sign}({num}/{den})"

    def _replace_digit_frac(m: re.Match) -> str:
        sign = m.group(1) or ""
        num = m.group(2)
        den = m.group(3)
        return f"{sign}({num}/{den})"

    text = re.sub(r"(-?)\\?(?:d|t)?frac\{([^}]+)\}\{([^}]+)\}", _replace_curly_frac, text)
    text = re.sub(r"(-?)\\?(?:d|t)?frac(\d)(\d)", _replace_digit_frac, text)
    text = re.sub(r"\\(?:d|t)?frac\b", "", text)

    # 5. Roots / Radicals
    text = re.sub(r"\\sqrt\[3\]\{([^}]+)\}", r"∛(\1)", text)
    text = re.sub(r"\\sqrt\[4\]\{([^}]+)\}", r"∜(\1)", text)
    text = re.sub(r"\\sqrt\{([^}]+)\}", r"√(\1)", text)
    text = re.sub(r"\\sqrt", "√", text)

    # 6. Degrees & Superscripts
    text = re.sub(r"⁽\\?circ\)|(?:\^|\\)circ\b|\\circ\b", "°", text)

    text = re.sub(r"\(\s*°\s*\)", "°", text)
    text = re.sub(r"\^\s*\(?\s*°\s*\)?", "°", text)

    def _to_sup(match: re.Match) -> str:
        inner = match.group(1)
        converted = "".join(_SUPERSCRIPTS.get(ch, ch) for ch in inner)
        return converted if all(ch in _SUPERSCRIPTS for ch in inner) else f"^({inner})"

    text = re.sub(r"\^\{([^}]+)\}", _to_sup, text)
    for orig_char, sup_char in _SUPERSCRIPTS.items():
        text = text.replace(f"^{orig_char}", sup_char)

    # 7. Subscripts (e.g. x_1, x_0, x_n, x_i) — only match math subscripts, not snake_case identifiers
    def _to_sub(match: re.Match) -> str:
        inner = match.group(1)
        converted = "".join(_SUBSCRIPTS.get(ch, ch) for ch in inner)
        return converted if all(ch in _SUBSCRIPTS for ch in inner) else f"_({inner})"

    text = re.sub(r"_\{([^}]+)\}", _to_sub, text)

    def _sub_single(match: re.Match) -> str:
        var = match.group(1)
        ch = match.group(2)
        return f"{var}{_SUBSCRIPTS.get(ch, '_' + ch)}"

    text = re.sub(r"([a-zA-Z0-9])_([0-9nixa-z])(?![a-zA-Z0-9])", _sub_single, text)

    # 8. Greek letters
    for pattern, sym in _GREEK.items():
        text = re.sub(re.escape(pattern) + r"(?![a-zA-Z])", sym, text)

    # 9. Operators and Relation Symbols
    for pattern, sym in _OPERATORS:
        text = re.sub(pattern, sym, text)

    # 10. Overlines / Bars (e.g. repeating decimals or line segments)
    text = re.sub(r"\\overline\{([^}]+)\}", r"\1(bar)", text)
    text = re.sub(r"\\bar\{([^}]+)\}", r"\1(bar)", text)

    # 11. Alignment & TeX space cleanup
    text = re.sub(r"&=(?:\s*)", "= ", text)
    text = re.sub(r"&(?:\s*)", "", text)
    text = re.sub(r"\\\\", "\n", text)
    text = re.sub(r"\\[;,:\!]|\\quad|\\qquad|\\ ", " ", text)

    # 12. Dollar signs ($...$ and $$...$$)
    text = re.sub(r"\$\$(.*?)\$\$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\$([^$\n]+)\$", r"\1", text)
    text = text.replace("$", "")

    # 13. Safety net: Remove remaining backslashes before words or isolated backslashes
    text = re.sub(r"\\([a-zA-Z]+)", r"\1", text)
    text = text.replace("\\", "")

    # 14. Clean up whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # 15. Deterministic Geometry Diagram Interceptor Engine
    from math_teacher.geometry.svg_engine import enforce_deterministic_svg_diagrams
    text = enforce_deterministic_svg_diagrams(text)

    return text

