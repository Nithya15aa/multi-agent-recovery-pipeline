import re
import math

# Calculator tool for GSM8K.
# Evaluates <calc>...</calc> tags and fixes arithmetic errors in reasoning steps.

_SAFE_MATH = {
    "abs": abs, "round": round, "min": min, "max": max,
    "int": int, "float": float,
    "sqrt": math.sqrt, "pow": pow,
    "pi": math.pi, "e": math.e,
    "ceil": math.ceil, "floor": math.floor,
    "log": math.log, "log10": math.log10,
}


def safe_eval(expression: str) -> str:
    """Evaluate a math expression safely. Returns result or error string."""
    expr = expression.strip()
    # Security: only allow math characters
    if re.search(r"[^\d\s\+\-\*/\.\(\),a-z_]", expr, re.I):
        return None
    try:
        result = eval(expr, {"__builtins__": {}}, _SAFE_MATH)
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        if isinstance(result, float):
            return f"{result:.6f}".rstrip("0").rstrip(".")
        return str(result)
    except Exception:
        return None


def evaluate_calc_tags(text: str) -> str:
    """Strategy B: Replace <calc>expr</calc> with computed results."""
    def _replace(match):
        expr = match.group(1)
        result = safe_eval(expr)
        return result if result else match.group(0)
    return re.sub(r"<calc>(.*?)</calc>", _replace, text, flags=re.DOTALL)


def verify_arithmetic_in_text(text: str) -> str:
    """
    Strategy A: Find patterns like "X * Y = Z" and recompute Z.
    If Z is wrong, replace with correct value.

    Catches common patterns:
      "15 * 3 = 44"  ->  "15 * 3 = 45"
      "100 + 250 = 300"  ->  "100 + 250 = 350"
      "Step 3: 45 - 9 = 34"  ->  "Step 3: 45 - 9 = 36"
    """
    pattern = r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)'

    def _fix(match):
        a, op, b, claimed = match.group(1), match.group(2), match.group(3), match.group(4)
        expr = f"{a} {op} {b}"
        correct = safe_eval(expr)
        if correct and correct != claimed:
            return f"{a} {op} {b} = {correct}"
        return match.group(0)

    return re.sub(pattern, _fix, text)


def apply_tools(text: str, domain: str = "gsm8k") -> str:
    """
    Combined tool application: first evaluate <calc> tags,
    then fix any remaining arithmetic errors.
    Only applied for the gsm8k domain.
    """
    if domain != "gsm8k":
        return text

    original = text

    # Strategy B: evaluate explicit <calc> tags
    text = evaluate_calc_tags(text)

    # Strategy A: fix any remaining arithmetic errors
    text = verify_arithmetic_in_text(text)

    if text != original:
        n_calcs = len(re.findall(r"<calc>", original))
        n_fixes = sum(1 for a, b in zip(original.split('\n'), text.split('\n')) if a != b)
        print(f"  [TOOL] Calculator: {n_calcs} <calc> tags evaluated, {n_fixes} lines corrected")

    return text


GSM8K_PROMPT_WITH_TOOLS = (
    "You are a math reasoning assistant with a calculator.\n"
    "Solve step by step (Step 1, Step 2, ...).\n"
    "IMPORTANT: For every arithmetic operation, write it as:\n"
    "  <calc>expression</calc>\n"
    "Examples:\n"
    "  Total = <calc>15 * 3</calc>\n"
    "  Remaining = <calc>100 - 45</calc>\n"
    "  Each person gets = <calc>120 / 4</calc>\n"
    "The calculator will compute the result automatically.\n"
    "State ONLY the final numeric answer on the last line.\n"
    "No units, currency symbols, or commas.\n\n"
    "{rag_block}Problem: {task}\n{feedback_block}"
)

SURGICAL_PROMPT_WITH_TOOLS = (
    "You are a recovery agent with a calculator. "
    "The steps below are VERIFIED CORRECT -- do NOT change them.\n"
    "Recompute from Step {fail_idx}.\n"
    "Use <calc>expression</calc> for ALL arithmetic.\n\n"
    "Problem: {task}\n\n"
    "=== Verified Correct Steps ===\n{prefix_text}\n\n"
    "Continue from Step {fail_idx}. Show your work.\n"
    "State ONLY the final numeric answer on the last line."
)


class ExecutionAgentWithTools:
    """
    Drop-in replacement for ExecutionAgent in agents.py.
    Includes calculator tool support for GSM8K reasoning.

    Usage:
        pipe.executor = ExecutionAgentWithTools(llm, domain, rag_fn)
    """

    PROMPTS = {
        "gsm8k": GSM8K_PROMPT_WITH_TOOLS,
        "arc": (
            "You are a science reasoning assistant.\n"
            "Show reasoning step by step.  Answer with ONLY the letter "
            "(A, B, C, or D) on the last line.\n\n"
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

        # Apply calculator tool: evaluates <calc> tags and fixes arithmetic errors
        return apply_tools(raw_output, self.domain)


if __name__ == "__main__":
    # Test Strategy B: <calc> tags
    text_b = """Step 1: John has 15 apples at $3 each.
Total = <calc>15 * 3</calc>
Step 2: He gets a 20% discount.
Discount = <calc>45 * 0.20</calc>
Step 3: Final price = <calc>45 - 9</calc>
36"""
    print("=== Strategy B (calc tags) ===")
    print(evaluate_calc_tags(text_b))
    print()

    # Test Strategy A: fix arithmetic errors (no tags needed)
    text_a = """Step 1: 15 * 3 = 44
Step 2: 44 - 9 = 36
Step 3: 36 + 10 = 45
45"""
    print("=== Strategy A (auto-fix) ===")
    print(verify_arithmetic_in_text(text_a))
    print()

    # Test combined
    text_c = """Step 1: Base cost = <calc>25 * 4</calc>
Step 2: Tax = 100 * 0.08 = 9
Step 3: Total = <calc>100 + 9</calc>
109"""
    print("=== Combined (both strategies) ===")
    print(apply_tools(text_c, "gsm8k"))