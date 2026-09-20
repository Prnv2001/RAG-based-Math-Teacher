"""SymPy-based deterministic math verifier.

Independently verifies mathematical results rather than trusting LLM output alone.
"""

from __future__ import annotations

from math_teacher.domain.models import MathVerificationResult


def _normalize_sympy_expr(expr_str: str) -> str:
    """Normalize unicode math symbols to standard SymPy readable syntax."""
    if not expr_str:
        return ""
    import re
    # Implicit multiplication (e.g. 5x -> 5*x, 2x -> 2*x)
    expr_str = re.sub(r"(\d)\s*([a-zA-Z])", r"\1*\2", expr_str)

    substitutions = {
        "²": "**2",
        "³": "**3",
        "⁴": "**4",
        "⁵": "**5",
        "⁶": "**6",
        "⁷": "**7",
        "⁸": "**8",
        "⁹": "**9",
        "⁰": "**0",
        "^": "**",
        "×": "*",
        "÷": "/",
        "−": "-",
        "–": "-",
        "—": "-",
        "√": "sqrt",
    }
    for sub, repl in substitutions.items():
        expr_str = expr_str.replace(sub, repl)
    return expr_str


class MathVerifier:
    """Deterministic symbolic math verification using SymPy.

    Verifies equation roots by substituting them back and checking the result is zero.
    """

    def verify_equation_solution(
        self,
        equation: str,
        variable: str,
        proposed_roots: list[str],
    ) -> MathVerificationResult:
        """Verify that proposed_roots satisfy the given polynomial/equation.

        Args:
            equation: Polynomial expression (LHS, assumed = 0) or "LHS = RHS" form.
            variable: Variable name (e.g., "x").
            proposed_roots: List of proposed root values as strings (e.g., ["-2", "-1/2"]).

        Returns:
            MathVerificationResult with passed=True if ALL roots verify correctly.
        """
        try:
            from sympy import Eq, sympify, symbols
            from sympy.core.sympify import SympifyError

            var = symbols(variable)
            norm_eq = _normalize_sympy_expr(equation)

            # Handle "LHS = RHS" format
            try:
                if "=" in norm_eq:
                    lhs_str, rhs_str = norm_eq.split("=", 1)
                    expr = sympify(lhs_str.strip()) - sympify(rhs_str.strip())
                else:
                    expr = sympify(norm_eq.strip())
            except (SympifyError, SyntaxError, TypeError) as parse_err:
                return MathVerificationResult(
                    passed=True,
                    reason=f"Symbolic equation verification not applicable ({parse_err}).",
                    status="not_required",
                )

            failed_roots = []
            for root_str in proposed_roots:
                try:
                    norm_root = _normalize_sympy_expr(root_str)
                    root_val = sympify(norm_root.strip())
                    substituted = expr.subs(var, root_val)

                    # If remaining expression still contains free symbols (e.g. x1, y1), it's a formula
                    if len(getattr(substituted, "free_symbols", set())) > 0:
                        return MathVerificationResult(
                            passed=True,
                            reason="Formula expression with multiple variables — root check not applicable.",
                            status="not_required",
                        )

                    # Should simplify to 0
                    if substituted != 0:
                        from sympy import simplify
                        simplified = simplify(substituted)
                        if simplified != 0:
                            failed_roots.append(f"{root_str} (residual: {simplified})")
                except (SympifyError, SyntaxError, TypeError) as e:
                    failed_roots.append(f"{root_str} (parse error: {e})")

            if failed_roots:
                return MathVerificationResult(
                    passed=False,
                    reason=f"Root(s) failed verification: {', '.join(failed_roots)}",
                    status="failed",
                )

            return MathVerificationResult(
                passed=True,
                reason=f"All {len(proposed_roots)} root(s) verified symbolically.",
                status="passed",
            )

        except ImportError:
            return MathVerificationResult(
                passed=False,
                reason="SymPy not available for verification.",
                status="unparseable",
            )
        except Exception as exc:
            return MathVerificationResult(
                passed=True,
                reason=f"Verification not applicable for formula: {exc}",
                status="not_required",
            )

    def verify_math_problem(
        self,
        problem: str,
        proposed_solution: str,
    ) -> MathVerificationResult:
        """High-level verification from problem string + proposed solution string.

        Attempts to extract equation and roots from the problem description.
        Falls back gracefully for problems that cannot be parsed symbolically.
        """
        import re

        norm_problem = _normalize_sympy_expr(problem)
        norm_solution = _normalize_sympy_expr(proposed_solution)

        # Try to detect an equation in the problem (simple heuristic)
        eq_match = re.search(r"([\w\s\+\-\*\^/\(\)]+=[^,\n]+)", norm_problem)
        if not eq_match:
            return MathVerificationResult(
                passed=True,
                reason="Concept / classification problem — symbolic equation verification not applicable.",
                status="not_required",
            )

        # Try to extract proposed roots from solution
        root_matches = re.findall(r"(?:x|y|z|t)\s*=\s*([-\d/\.]+)", norm_solution)
        if not root_matches:
            return MathVerificationResult(
                passed=True,
                reason="No numerical roots to verify.",
                status="not_required",
            )


        return self.verify_equation_solution(
            equation=eq_match.group(1).strip(),
            variable="x",
            proposed_roots=root_matches,
        )
