"""
Tool-Augmented Execution Agent
==============================
Adds a Python code interpreter / calculator tool to the Execution Agent.
Instead of relying on the LLM to compute arithmetic, the agent can
emit <calc>expression</calc> tags which are evaluated locally.

This eliminates most ARITHMETIC_ERROR failures while preserving
the LLM's reasoning structure. Non-arithmetic errors (MISREAD_PROBLEM,
LOGIC_GAP, KNOWLEDGE_GAP) are unaffected.


"""

import re
import math


#safe math functions for eval, to prevent security issues while allowing basic arithmetic and common math functions
_SAFE_MATH = {
    "abs": abs, "round": round, "min": min, "max": max,
    "int": int, "float": float,
    "sqrt": math.sqrt, "pow": pow,
    "pi": math.pi, "e": math.e,
    "ceil": math.ceil, "floor": math.floor,
    "log": math.log, "log10": math.log10,
}


def safe_eval(expression: str) -> str:
    """
    Evaluate a math expression safely.
    Returns the result as a string, or the original expression on error.
    """
    expr = expression.strip()
    # Remove any non-math characters for safety
    # Allow: digits, operators, parens, dots, commas, spaces, function names
    if re.search(r"[^\d\s\+\-\*/\.\(\),a-z_]", expr, re.I):
        return f"[ERROR: unsafe expression: {expr}]"

    try:
        result = eval(expr, {"__builtins__": {}}, _SAFE_MATH)
        # Format nicely
        if isinstance(result, float):
            if result == int(result):
                return str(int(result))
            return f"{result:.6f}".rstrip("0").rstrip(".")
        return str(result)
    except Exception as e:
        return f"[ERROR: {e}]"


def calculate_expressions(text: str) -> str:
    """
    Find all <calc>...</calc> tags in text, evaluate them,
    and replace with results.

    Example:
        Input:  "The total is <calc>247 * 13</calc> dollars"
        Output: "The total is 3211 dollars"
    """
    def _replace(match):
        expr = match.group(1)
        result = safe_eval(expr)
        return result

    return re.sub(r"<calc>(.*?)</calc>", _replace, text, flags=re.DOTALL)


#tool-augmented execution agent that instructs the LLM to use <calc> tags for arithmetic,
# and post-processes output to evaluate those tags
class ToolAugmentedExecutionAgent:
    

    PROMPTS = {
        "gsm8k": (
            "You are a math reasoning assistant with a calculator tool.\n"
            "Solve step by step (Step 1, Step 2, …).\n"
            "For ANY arithmetic computation, wrap it in <calc> tags.\n"
            "Example: The cost is <calc>15 * 3 + 7</calc> = 52\n"
            "The calculator will evaluate expressions automatically.\n"
            "State ONLY the final numeric answer on the last line.\n"
            "No units, currency symbols, or commas.\n\n"
            "{rag_block}Problem: {task}\n{feedback_block}"
        ),
        "arc": (
            "You are a science reasoning assistant.\n"
            "Show reasoning step by step. For any calculations, use "
            "<calc>expression</calc> tags.\n"
            "Answer with ONLY the letter (A, B, C, or D) on the last line.\n\n"
            "{rag_block}=== Question ===\n{task}\n{feedback_block}"
        ),
        "boolq": (
            "You are a reading comprehension assistant.\n"
            "Show reasoning step by step.  Answer with ONLY 'yes' or 'no' "
            "on the last line.\n\n"
            "{rag_block}=== Question ===\n{task}\n{feedback_block}"
        ),
    }

    def __init__(self, llm, domain, rag_fn=None):
        self.llm = llm
        self.domain = domain
        self.rag_fn = rag_fn

    def execute(self, task, feedback="", temperature=None):
        tmpl = self.PROMPTS.get(self.domain, self.PROMPTS["gsm8k"])
        rag = ""
        if self.rag_fn:
            rag = f"=== Relevant Examples ===\n{self.rag_fn(task, top_k=3)}\n\n"
        fb = ""
        if feedback:
            fb = f"=== Feedback from previous attempt ===\n{feedback}\n"

        raw_output = self.llm.ask(
            tmpl.format(task=task, rag_block=rag, feedback_block=fb),
            temperature=temperature,
        )

        # Post-process: evaluate any <calc> tags
        processed = calculate_expressions(raw_output)

        return processed

# Merged verification and classification agent that performs both tasks in a single LLM call,
#  with a structured prompt and response format to extract the necessary information for both steps at once.
class MergedVerifyClassifyAgent:
    

    PROMPT_TEMPLATE = (
        "You are a verification and classification agent.\n"
        "Analyze the following problem and model output.\n\n"
        "Problem: {task}\n"
        "Model Output: {output}\n"
        "{gt_block}"
        "\n"
        "Respond in EXACTLY this format:\n"
        "VERDICT: VALID or INVALID\n"
        "CONFIDENCE: HIGH or LOW\n"
        "REASON: <one sentence explanation>\n"
        "FAILURE_TYPE: <one of: ARITHMETIC_ERROR, MISREAD_PROBLEM, "
        "MISSING_CONSTRAINT, LOGIC_GAP, KNOWLEDGE_GAP, NONE>\n"
    )

    def __init__(self, llm, domain, taxonomy_labels=None):
        self.llm = llm
        self.domain = domain
        self.taxonomy_labels = taxonomy_labels or [
            "ARITHMETIC_ERROR", "MISREAD_PROBLEM",
            "MISSING_CONSTRAINT", "LOGIC_GAP", "KNOWLEDGE_GAP",
        ]

    def verify_and_classify(self, task, output, ground_truth=None, use_gt=True):
        """
        Single call that returns (flagged, reason, confidence, failure_type).
        """
        # Deterministic check first (no API call needed)
        if ground_truth is not None and use_gt:
            norm_out = self._normalize(output)
            norm_gt = self._normalize_gt(ground_truth)
            if norm_out == norm_gt:
                return False, "Deterministic: exact match", "CONFIDENT", "NONE"
            # Still need LLM to classify the failure type
            gt_block = f"Expected Answer: {ground_truth}\n"
        else:
            gt_block = ""

        prompt = self.PROMPT_TEMPLATE.format(
            task=task[:500], output=output, gt_block=gt_block
        )
        raw = self.llm.ask(prompt)

        # Parse response
        flagged = "INVALID" in raw.upper()
        conf = "CONFIDENT" if "HIGH" in raw.upper() else "UNCERTAIN"
        reason = ""
        ftype = "LOGIC_GAP"  # default

        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("REASON:"):
                reason = line[7:].strip()[:200]
            if line.startswith("FAILURE_TYPE:"):
                ft_raw = line[13:].strip().upper().replace(" ", "_")
                for label in self.taxonomy_labels:
                    if label in ft_raw:
                        ftype = label
                        break

        if not flagged:
            ftype = "NONE"

        return flagged, reason, conf, ftype

    def _normalize(self, t):
        from .agents import normalize_numeric, normalize_letter, normalize_yesno
        if self.domain == "gsm8k":  return normalize_numeric(t)
        if self.domain == "arc":    return normalize_letter(t)
        return normalize_yesno(t)

    def _normalize_gt(self, t):
        from .agents import normalize_numeric
        if self.domain == "gsm8k":  return normalize_numeric(t)
        if self.domain == "arc":    return t.strip().upper()
        return t.strip().lower()


