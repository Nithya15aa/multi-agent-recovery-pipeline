"""
Enhanced Multi-Agent Failure Recovery Pipeline
6-Agent Architecture with Step-Level Attribution,
Recoverability Estimation, and Self-Mutating Failure Ontology


"""

import re
import time
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .tool_routing import apply_tools

#llm client factory with dual SDK support
def create_gemini_client(api_key: str):
  
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client, "genai"
    except ImportError:
        pass
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai, "generativeai"
    except ImportError:
        pass
    raise ImportError(
        "No Gemini SDK found.  Install one of:\n"
        "  pip install google-genai            (recommended)\n"
        "  pip install google-generativeai      (alternative)\n"
    )


#llm backend wrapper with retry logic and token counting
class LLMBackend:
   

    def __init__(self, client, model="gemini-2.5-flash",
                 retries=3, base_wait=5.0, inter_call_delay=4.0,
                 sdk_type="genai"):
        self.client = client
        self.model = model
        self.retries = retries
        self.base_wait = base_wait
        self.inter_call_delay = inter_call_delay
        self.sdk_type = sdk_type
        self.total_calls = 0
        self.total_tokens_approx = 0

    def ask(self, prompt, temperature=None):
        for attempt in range(self.retries):
            try:
                if self.sdk_type == "genai":
                    kw = {"model": self.model, "contents": prompt}
                    if temperature is not None:
                        kw["config"] = {"temperature": temperature}
                    resp = self.client.models.generate_content(**kw)
                    text = resp.text.strip()
                else:
                    mdl = self.client.GenerativeModel(self.model)
                    gc = {"temperature": temperature} if temperature else None
                    resp = mdl.generate_content(prompt, generation_config=gc)
                    text = resp.text.strip()
                self.total_calls += 1
                self.total_tokens_approx += len(text.split()) + len(prompt.split())
                return text
            except Exception as e:
                print(f"  [API error {attempt+1}/{self.retries}]: {e}")
                if attempt < self.retries - 1:
                    time.sleep(self.base_wait * (attempt + 1))
        return ""

    def delay(self):
        time.sleep(self.inter_call_delay)


#dynamic failure taxonomy with optional self-mutation based on clustering of failure contexts and outcomes
class FailureTaxonomy:
    BASE_TYPES = [
        "ARITHMETIC_ERROR", "MISREAD_PROBLEM",
        "MISSING_CONSTRAINT", "LOGIC_GAP", "KNOWLEDGE_GAP",
    ]

    def __init__(self, enable_evolution=False, split_threshold=0.3,
                 min_samples_for_split=6, max_categories=10):
        self.categories = list(self.BASE_TYPES)
        self.enable_evolution = enable_evolution
        self.split_threshold = split_threshold
        self.min_samples_for_split = min_samples_for_split
        self.max_categories = max_categories
        self.evolution_log = []
        self._cat_outcomes = defaultdict(list)

    @property
    def labels(self):
        return list(self.categories)

    def record_outcome(self, cat, context, recovered):
        self._cat_outcomes[cat].append((context, recovered))

    def maybe_evolve(self, vectorizer):
        if not self.enable_evolution or len(self.categories) >= self.max_categories:
            return
        for cat in list(self.categories):
            outcomes = self._cat_outcomes.get(cat, [])
            if len(outcomes) < self.min_samples_for_split:
                continue
            ok = [o for o in outcomes if o[1]]
            fail = [o for o in outcomes if not o[1]]
            if not ok or not fail:
                continue
            rsr = len(ok) / len(outcomes)
            if not (0.15 < rsr < 0.85):
                continue
            try:
                from sklearn.cluster import AgglomerativeClustering
                vecs = vectorizer.transform([o[0] for o in outcomes])
                labs = [1 if o[1] else 0 for o in outcomes]
                cl = AgglomerativeClustering(n_clusters=2, metric="cosine",
                                             linkage="average")
                cl_labels = cl.fit_predict(vecs.toarray())
                c0 = [labs[i] for i in range(len(labs)) if cl_labels[i] == 0]
                c1 = [labs[i] for i in range(len(labs)) if cl_labels[i] == 1]
                if c0 and c1 and abs(sum(c0)/len(c0) - sum(c1)/len(c1)) > self.split_threshold:
                    a, b = f"{cat}_A", f"{cat}_B"
                    self.categories.remove(cat)
                    self.categories.extend([a, b])
                    self._cat_outcomes[a] = [outcomes[i] for i in range(len(outcomes)) if cl_labels[i] == 0]
                    self._cat_outcomes[b] = [outcomes[i] for i in range(len(outcomes)) if cl_labels[i] == 1]
                    del self._cat_outcomes[cat]
                    self.evolution_log.append({"action": "split", "from": cat, "to": [a, b]})
                    print(f"  [SMFO] Split {cat} → {a} + {b}")
            except Exception:
                pass


#failure memory for RAG-augmented recovery, storing past failure contexts 
# and outcomes for retrieval based on similarity to new failures
class FailureMemory:
    def __init__(self, vectorizer):
        self.entries = []
        self.vectorizer = vectorizer

    def add(self, question, wrong_answer, failure_type, reason,
            correct_answer=None, failure_step=None, reasoning_prefix=None):
        self.entries.append(dict(
            question=question, wrong_answer=wrong_answer,
            failure_type=failure_type, reason=reason,
            correct_answer=correct_answer,
            failure_step=failure_step, reasoning_prefix=reasoning_prefix,
        ))

    def retrieve(self, query, top_k=2):
        if not self.entries:
            return ""
        fm = self.vectorizer.transform([e["question"] for e in self.entries])
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, fm).flatten()
        idxs = np.argsort(sims)[::-1][:top_k]
        blocks = []
        for i in idxs:
            f = self.entries[i]
            blk = (f"Past Failure [{f['failure_type']}]:\n"
                   f"Problem   : {f['question'][:200]}\n"
                   f"Wrong Ans : {f['wrong_answer']}\n"
                   f"Why Wrong : {f['reason']}\n")
            if f.get("correct_answer"):
                blk += f"Correct   : {f['correct_answer']}\n"
            if f.get("failure_step") is not None:
                blk += f"Error at step: {f['failure_step']}\n"
            blocks.append(blk)
        return "\n\n---\n\n".join(blocks)

    @property
    def size(self):
        return len(self.entries)

    def clear(self):
        self.entries.clear()

#recoverability estimator using a simple Beta distribution per failure type and domain, 
# updated based on recovery outcomes, to guide retry decisions

class RecoverabilityEstimator:
    def __init__(self, domain_priors=None):
        self._c = defaultdict(lambda: {"a": 1.0, "b": 1.0})
        if domain_priors:
            for key, p in domain_priors.items():
                self._c[key]["a"] = p * 5 + 1
                self._c[key]["b"] = (1 - p) * 5 + 1

    def estimate(self, ftype, domain):
        c = self._c[(ftype, domain)]
        return c["a"] / (c["a"] + c["b"])

    def update(self, ftype, domain, recovered):
        k = (ftype, domain)
        if recovered:
            self._c[k]["a"] += 1
        else:
            self._c[k]["b"] += 1

    def recommend_retries(self, ftype, domain):
        p = self.estimate(ftype, domain)
        if p > 0.7: return 2
        if p > 0.3: return 1
        return 0

    def get_all_estimates(self):
        return {str(k): round(v["a"]/(v["a"]+v["b"]), 3) for k, v in self._c.items()}


#step attributor to identify divergence points in reasoning chains between multiple LLM outputs,
#  to enable targeted "surgical" prompting for recovery from specific step errors
class StepAttributor:
    @staticmethod
    def parse_steps(reasoning):
        lines = reasoning.strip().split("\n")
        steps, cur = [], []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            if re.match(r"^(Step\s+\d+|[\d]+[\.\):]|\*\*Step)", ln, re.I):
                if cur:
                    steps.append(" ".join(cur))
                cur = [ln]
            else:
                cur.append(ln)
        if cur:
            steps.append(" ".join(cur))
        if len(steps) <= 1:
            sents = re.split(r"(?<=[.!?])\s+", reasoning.strip())
            steps = [s for s in sents if len(s.strip()) > 10]
        return steps

    @staticmethod
    def find_divergence(steps_a, steps_b, vectorizer, threshold=0.5):
        n = min(len(steps_a), len(steps_b))
        if n == 0:
            return 0
        for i in range(n):
            try:
                vecs = vectorizer.transform([steps_a[i], steps_b[i]])
                sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
                if sim < threshold:
                    return i
            except Exception:
                return i
        return -1

    @staticmethod
    def build_surgical_prompt(task, verified_prefix, fail_idx, domain="gsm8k"):
        prefix_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(verified_prefix))
        inst = {
            "gsm8k": (
                "Use <calc>expression</calc> for ALL arithmetic.\n"
                "State ONLY the final numeric answer on the last line. "
                "No units, currency, or commas."
            ),
            "arc":   "Answer with ONLY the letter (A, B, C, or D) on the last line.",
            "boolq": "Answer with ONLY 'yes' or 'no' on the last line.",
        }.get(domain, "")
        return (
            f"You are a recovery agent. The steps below are VERIFIED CORRECT — "
            f"do NOT change them.  Recompute from Step {fail_idx+1}.\n\n"
            f"Problem: {task}\n\n"
            f"=== Verified Correct Steps ===\n{prefix_text}\n\n"
            f"Continue from Step {fail_idx+1}. Show your work.\n{inst}"
        )


#normalization functions to standardize LLM outputs for verification and comparison
def normalize_numeric(text):
    text = text.strip().replace("$", "").replace(",", "").replace("%", "")
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else text.strip()

def normalize_letter(text):
    text = text.strip().upper()
    letters = re.findall(r"\b[A-D]\b", text)
    return letters[-1] if letters else text[:1]

def normalize_yesno(text):
    t = text.strip().lower()
    if t.startswith("yes"): return "yes"
    if t.startswith("no"):  return "no"
    if "yes" in t: return "yes"
    if "no" in t:  return "no"
    return t[:3]

#execution agent to generate answers with optional RAG context and feedback from previous attempts,
class ExecutionAgent:
    # ── CHANGE 4: Updated GSM8K prompt with <calc> instructions ──
    PROMPTS = {
        "gsm8k": (
            "You are a math reasoning assistant with a calculator.\n"
            "Solve step by step (Step 1, Step 2, …).\n"
            "For EVERY arithmetic operation, use <calc>expression</calc>.\n"
            "Examples:\n"
            "  Total = <calc>15 * 3</calc>\n"
            "  Remaining = <calc>100 - 45</calc>\n"
            "The calculator will compute the result automatically.\n"
            "State ONLY the final numeric answer on the last line.\n"
            "No units, currency symbols, or commas.\n\n"
            "{rag_block}Problem: {task}\n{feedback_block}"
        ),
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
        self.llm, self.domain, self.rag_fn = llm, domain, rag_fn

    def execute(self, task, feedback="", temperature=None):
        tmpl = self.PROMPTS.get(self.domain, self.PROMPTS["gsm8k"])
        rag = f"=== Relevant Examples ===\n{self.rag_fn(task, top_k=3)}\n\n" if self.rag_fn else ""
        fb = f"=== Feedback from previous attempt ===\n{feedback}\n" if feedback else ""
        raw = self.llm.ask(tmpl.format(task=task, rag_block=rag, feedback_block=fb),
                           temperature=temperature)
        # ── CHANGE 2: Apply calculator tool to LLM output ──
        return apply_tools(raw, self.domain)


class VerificationAgent:
    def __init__(self, llm, domain):
        self.llm, self.domain = llm, domain

    def verify(self, task, output, ground_truth=None, use_gt=True):
        norm = self._norm(output)
        if ground_truth is not None and use_gt:
            gt = self._norm_gt(ground_truth)
            if norm == gt:
                return False, "Deterministic: exact match", "CONFIDENT"
            return True, f"Deterministic: got {norm}, expected {gt}", "CONFIDENT"
        # LLM critique
        prompt = (
            f"You are a verification agent.\n"
            f"Problem: {task[:500]}\nModel Output: {output}\n\n"
            f"Respond ONLY with:\nVERDICT: VALID or INVALID\n"
            f"CONFIDENCE: HIGH or LOW\nREASON: <one sentence>"
        )
        raw = self.llm.ask(prompt)
        flagged = "INVALID" in raw.upper()
        conf = "CONFIDENT" if "HIGH" in raw.upper() else "UNCERTAIN"
        return flagged, raw[:200], conf

    def _norm(self, t):
        if self.domain == "gsm8k":  return normalize_numeric(t)
        if self.domain == "arc":    return normalize_letter(t)
        return normalize_yesno(t)

    def _norm_gt(self, t):
        if self.domain == "gsm8k":  return normalize_numeric(t)
        if self.domain == "arc":    return t.strip().upper()
        return t.strip().lower()


class ClassifierAgent:
    def __init__(self, llm, taxonomy):
        self.llm, self.taxonomy = llm, taxonomy

    def classify(self, task, output, reason):
        labels = self.taxonomy.labels
        prompt = (
            f"Classify this failure into exactly one category.\n\n"
            f"Problem     : {task[:400]}\nWrong Output: {output}\n"
            f"Failure Note: {reason}\n\nCategories:\n"
            + "\n".join(f"- {l}" for l in labels)
            + "\n\nRespond with ONLY the category name."
        )
        raw = self.llm.ask(prompt).strip().upper().replace(" ", "_")
        for l in labels:
            if l in raw:
                return l
        return "LOGIC_GAP"


class RecoveryAgent:
    TEMPLATES = {
        "ARITHMETIC_ERROR":  "The model made a CALCULATION MISTAKE.\n",
        "MISREAD_PROBLEM":   "The model MISUNDERSTOOD THE QUESTION.\n",
        "MISSING_CONSTRAINT":"The model IGNORED A CONSTRAINT.\n",
        "LOGIC_GAP":         "The model had a FLAWED REASONING CHAIN.\n",
        "KNOWLEDGE_GAP":     "The model may lack required knowledge.\n",
    }
    ANSWER_INST = {
        "gsm8k": (
            "Use <calc>expression</calc> for ALL arithmetic.\n"
            "State ONLY the final numeric answer on the last line."
        ),
        "arc":   "Answer with ONLY the letter (A, B, C, or D) on the last line.",
        "boolq": "Answer with ONLY 'yes' or 'no' on the last line.",
    }

    def __init__(self, llm, domain, memory):
        self.llm, self.domain, self.memory = llm, domain, memory

    def recover(self, task, output, reason, ftype, surgical_prompt=None):
        if surgical_prompt:
            past = self.memory.retrieve(task, top_k=2)
            if past:
                surgical_prompt += f"\n\n=== Similar Past Failures ===\n{past}\n"
            raw = self.llm.ask(surgical_prompt)
            # ── CHANGE 3: Apply calculator tool to recovery output ──
            return apply_tools(raw, self.domain)
        base = ftype.split("_A")[0].split("_B")[0]
        desc = self.TEMPLATES.get(base, self.TEMPLATES["LOGIC_GAP"])
        inst = self.ANSWER_INST.get(self.domain, "")
        prompt = (
            f"You are a recovery agent. {desc}"
            f"Problem: {task}\nWrong Output: {output}\n"
            f"Feedback: {reason}\nRebuild reasoning step by step.\n{inst}"
        )
        past = self.memory.retrieve(task, top_k=2)
        if past:
            prompt += f"\n\n=== Similar Past Failures ===\n{past}\n"
        raw = self.llm.ask(prompt)
        # ── CHANGE 3: Apply calculator tool to recovery output ──
        return apply_tools(raw, self.domain)

#pipeline orchestrator to manage the end-to-end process of execution, verification, classification, and recovery, 
# with support for multiple attempts, feedback loops, and logging of outcomes for analysis and taxonomy evolution
class PipelineOrchestrator:
    def __init__(self, llm, domain, vectorizer, rag_fn=None,
                 max_retries=2, enable_step_attribution=True,
                 enable_recoverability=True, enable_taxonomy_evolution=False,
                 enable_rag_memory=True, random_classification=False,
                 use_ground_truth_verification=True, evolution_interval=10):
        self.llm = llm
        self.domain = domain
        self.vectorizer = vectorizer
        self.max_retries = max_retries
        self.enable_step_attribution = enable_step_attribution
        self.enable_recoverability = enable_recoverability
        self.enable_rag_memory = enable_rag_memory
        self.random_classification = random_classification
        self.use_gt = use_ground_truth_verification
        self.evolution_interval = evolution_interval

        self.taxonomy = FailureTaxonomy(enable_evolution=enable_taxonomy_evolution)
        self.memory = FailureMemory(vectorizer)
        self.estimator = RecoverabilityEstimator(domain_priors={
            ("ARITHMETIC_ERROR", domain): 0.9,
            ("MISREAD_PROBLEM", domain): 0.7,
            ("MISSING_CONSTRAINT", domain): 0.5,
            ("LOGIC_GAP", domain): 0.4,
            ("KNOWLEDGE_GAP", domain): 0.1,
        })
        self.attributor = StepAttributor()
        self.executor = ExecutionAgent(llm, domain, rag_fn)
        self.verifier = VerificationAgent(llm, domain)
        self.classifier = ClassifierAgent(llm, self.taxonomy)
        self.recovery = RecoveryAgent(llm, domain, self.memory)
        self._fail_count = 0

    def normalize(self, text):
        if self.domain == "gsm8k":  return normalize_numeric(text)
        if self.domain == "arc":    return normalize_letter(text)
        return normalize_yesno(text)

    # baseline with no verification or recovery
    def run_baseline_a(self, task):
        return self.normalize(self.executor.execute(task))

    def run_baseline_b(self, task, gt=None):
        out = self.executor.execute(task)
        flagged, _, _ = self.verifier.verify(task, out, gt, self.use_gt)
        return self.normalize(out), flagged

    def run_majority_vote(self, task, k=3):
        from collections import Counter
        ans = []
        for _ in range(k):
            ans.append(self.normalize(self.executor.execute(task, temperature=0.7)))
            self.llm.delay()
        return Counter(ans).most_common(1)[0][0]

    # full pipeline with multi-agent interaction, feedback loops, and adaptive behavior based on failure analysis and recoverability estimation
    def run_pipeline(self, task, ground_truth=None):
        feedback = ""
        output = ""
        first_ans = None
        verdicts, ftypes, step_attrs, recov_scores = [], [], [], []
        surgical_used = False

        for attempt in range(self.max_retries + 1):
            output = self.executor.execute(task, feedback=feedback)
            if attempt == 0:
                first_ans = self.normalize(output)

            flagged, reason, conf = self.verifier.verify(task, output, ground_truth, self.use_gt)
            verdicts.append({"attempt": attempt+1, "flagged_invalid": flagged,
                             "reason": reason[:100], "confidence": conf})

            if not flagged:
                return self._result(output, attempt+1, first_ans, verdicts,
                                    ftypes, step_attrs, recov_scores, surgical_used)

            # Classify
            if self.random_classification:
                import random
                ftype = random.choice(self.taxonomy.labels)
            else:
                ftype = self.classifier.classify(task, output, reason)
            ftypes.append(ftype)
            self._fail_count += 1
            print(f"  Attempt {attempt+1} | Classified: {ftype}")

            # Recoverability
            if self.enable_recoverability:
                p = self.estimator.estimate(ftype, self.domain)
                recov_scores.append(p)
                if self.estimator.recommend_retries(ftype, self.domain) == 0 and attempt > 0:
                    print(f"  Recoverability: {p:.2f} → aborting")
                    break
            else:
                recov_scores.append(None)

            # Step attribution (first failure only)
            surgical_prompt = None
            if self.enable_step_attribution and attempt == 0:
                out2 = self.executor.execute(task, temperature=0.7)
                sa = self.attributor.parse_steps(output)
                sb = self.attributor.parse_steps(out2)
                if len(sa) >= 2 and len(sb) >= 2:
                    div = self.attributor.find_divergence(sa, sb, self.vectorizer)
                    if 0 <= div < len(sa):
                        surgical_prompt = self.attributor.build_surgical_prompt(
                            task, sa[:div], div, self.domain)
                        step_attrs.append({"step": div, "total": len(sa)})
                        surgical_used = True
                        print(f"  Step attribution: error at step {div+1}/{len(sa)}")

            # Store failure
            if self.enable_rag_memory:
                self.memory.add(task, self.normalize(output), ftype, reason,
                                ground_truth,
                                failure_step=step_attrs[-1]["step"] if step_attrs else None)

            # Recover
            feedback = self.recovery.recover(task, output, reason, ftype, surgical_prompt)

            # SMFO check
            if self._fail_count % self.evolution_interval == 0:
                self.taxonomy.maybe_evolve(self.vectorizer)
            self.llm.delay()

        return self._result(output, self.max_retries+1, first_ans, verdicts,
                            ftypes, step_attrs, recov_scores, surgical_used)

    def update_estimator(self, ftype, recovered):
        self.estimator.update(ftype, self.domain, recovered)
        self.taxonomy.record_outcome(ftype, "", recovered)

    def _result(self, output, attempts, first_ans, verdicts,
                ftypes, step_attrs, recov_scores, surgical_used):
        return {
            "output": self.normalize(output),
            "attempts": attempts,
            "first_attempt_ans": first_ans,
            "verdicts_log": verdicts,
            "failure_types": ftypes,
            "step_attributions": step_attrs,
            "recoverability_scores": recov_scores,
            "surgical_used": surgical_used,
            "memory_size": self.memory.size,
        }

    def get_state(self):
        return {
            "taxonomy": self.taxonomy.labels,
            "taxonomy_evolution_log": self.taxonomy.evolution_log,
            "memory_size": self.memory.size,
            "estimator_state": self.estimator.get_all_estimates(),
            "total_llm_calls": self.llm.total_calls,
            "approx_tokens": self.llm.total_tokens_approx,
        }