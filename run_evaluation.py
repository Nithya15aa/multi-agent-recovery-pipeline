#!/usr/bin/env python3
"""
Enhanced Multi-Agent Failure Recovery Pipeline — Runner

Usage:
  python run_evaluation.py --domain gsm8k --samples 50 --config full_pipeline --api-key YOUR_KEY
  python run_evaluation.py --domain arc   --samples 200 --all-ablations
  python run_evaluation.py --domain boolq --samples 50  --run-majority
"""

import argparse, json, os, sys, getpass


def check_deps():
    missing = []
    gemini_ok = False
    try:
        from google import genai; gemini_ok = True  # noqa
    except ImportError:
        pass
    if not gemini_ok:
        try:
            import google.generativeai; gemini_ok = True  # noqa
        except ImportError:
            pass
    if not gemini_ok:
        missing.append("google-genai")
    for mod, pkg in [("datasets","datasets"),("sklearn","scikit-learn"),("numpy","numpy")]:
        try: __import__(mod)
        except ImportError: missing.append(pkg)
    if missing:
        print("\nMissing packages:  " + ", ".join(missing))
        print("Fix:  pip install " + " ".join(missing) + "\n")
        sys.exit(1)


def main():
    check_deps()

    parser = argparse.ArgumentParser(description="Enhanced Pipeline Evaluation")
    parser.add_argument("--domain", choices=["gsm8k","arc","boolq"], default="gsm8k")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--config", default="full_pipeline")
    parser.add_argument("--all-ablations", action="store_true")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--delay", type=float, default=4.0)
    parser.add_argument("--run-majority", action="store_true")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter Gemini API key: ")
    if not api_key:
        print("Error: no API key."); sys.exit(1)

    # ── Make 'core' importable regardless of cwd ─────
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)

    core_dir = os.path.join(SCRIPT_DIR, "core")
    if not os.path.isdir(core_dir):
        print(f"\nCannot find  core/  next to this script.")
        print(f"Expected at: {core_dir}")
        print(f"\nMake sure your folder looks like:")
        print(f"  your_folder/")
        print(f"    run_evaluation.py")
        print(f"    core/")
        print(f"      __init__.py")
        print(f"      agents.py")
        print(f"      evaluation.py")
        sys.exit(1)

    import numpy as np
    from datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from core import (
        create_gemini_client, LLMBackend, EvaluationRunner,
        ABLATION_CONFIGS, print_metrics_table,
    )

    # connect to Gemini API and test model
    print("\nConnecting to Gemini API …")
    client, sdk_type = create_gemini_client(api_key)
    print(f"  SDK : {sdk_type}")
    llm = LLMBackend(client, model=args.model,
                     inter_call_delay=args.delay, sdk_type=sdk_type)

    print(f"  Testing model '{args.model}' …")
    test = llm.ask("Say hello.")
    if not test:
        print(f"  API call failed.  Check key / model name.")
        print(f"  Try:  --model gemini-2.0-flash  or  --model gemini-1.5-flash")
        sys.exit(1)
    print(f"  OK  ({test[:30]}…)")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print(f"\nLoading {args.domain} …")
    if args.domain == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="test")
    elif args.domain == "arc":
        dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    else:
        dataset = load_dataset("boolq", split="validation")
    print(f"  {len(dataset)} samples. Evaluating {min(args.samples, len(dataset))}.")

    # build RAG index (for few-shot retrieval in some configs)
    print("Building RAG index …")
    vectorizer = TfidfVectorizer(max_features=50000)
    rag_fn = None

    if args.domain == "arc":
        tr = load_dataset("ai2_arc", "ARC-Challenge", split="train")
        tq = []; ta = []
        for s in tr:
            cstr = "\n".join(f"{l}. {t}" for l,t in zip(s["choices"]["label"],s["choices"]["text"]))
            tq.append(f"{s['question']}\n{cstr}"); ta.append(f"Answer: {s['answerKey']}")
        mat = vectorizer.fit_transform(tq)
        def rag_fn(query, top_k=3):
            sims = cosine_similarity(vectorizer.transform([query]), mat).flatten()
            return "\n\n---\n\n".join(f"Example:\n{tq[i]}\n{ta[i]}" for i in np.argsort(sims)[::-1][:top_k])

    elif args.domain == "boolq":
        tr = load_dataset("boolq", split="train")
        tq = [f"Passage: {s['passage']}\nQuestion: {s['question']}" for s in tr]
        ta = ["Answer: yes" if s["answer"] else "Answer: no" for s in tr]
        mat = vectorizer.fit_transform(tq)
        def rag_fn(query, top_k=3):
            sims = cosine_similarity(vectorizer.transform([query]), mat).flatten()
            return "\n\n---\n\n".join(f"Example:\n{tq[i]}\n{ta[i]}" for i in np.argsort(sims)[::-1][:top_k])

    else:  # gsm8k
        tr = load_dataset("gsm8k", "main", split="train")
        tq = [s["question"] for s in tr]
        ta = [s["answer"].split("####")[-1].strip() for s in tr]
        mat = vectorizer.fit_transform(tq)
        def rag_fn(query, top_k=3):
            sims = cosine_similarity(vectorizer.transform([query]), mat).flatten()
            return "\n\n---\n\n".join(f"Example:\n{tq[i]}\nAnswer: {ta[i]}" for i in np.argsort(sims)[::-1][:top_k])

    print(f"  {len(tq)} training examples indexed.")

    # Run evaluation for each config
    cfgs = list(ABLATION_CONFIGS.keys()) if args.all_ablations else [args.config]
    all_m = {}

    for cname in cfgs:
        if cname not in ABLATION_CONFIGS:
            print(f"  Unknown config '{cname}'.  Available: {list(ABLATION_CONFIGS.keys())}")
            continue
        cfg = ABLATION_CONFIGS[cname]
        print(f"\n{'='*60}\n  {cname}  |  {args.domain}  |  {args.samples} samples\n{'='*60}")

        runner = EvaluationRunner(llm=llm, domain=args.domain,
                                  vectorizer=vectorizer, rag_fn=rag_fn, config_name=cname)
        m, _, _ = runner.run(
            dataset, args.samples,
            csv_path=os.path.join(args.output_dir, f"{args.domain}_{cname}_results.csv"),
            rsr_csv_path=os.path.join(args.output_dir, f"{args.domain}_{cname}_rsr.csv"),
            run_baselines=True, run_majority=args.run_majority, pipeline_kwargs=cfg)
        print_metrics_table(m)
        all_m[cname] = m

        with open(os.path.join(args.output_dir, f"{args.domain}_{cname}_metrics.json"), "w") as f:
            json.dump({k:v for k,v in m.items()
                       if isinstance(v,(str,int,float,bool,type(None),dict,list))},
                      f, indent=2, default=str)

    if len(all_m) > 1:
        print(f"\n{'='*80}\n{'ABLATION SUMMARY':^80}\n{'='*80}")
        print(f"  {'Config':<30} {'TSR':>8} {'RSR':>8} {'AAT':>8}")
        print(f"  {'-'*54}")
        for nm, mt in sorted(all_m.items(), key=lambda x:-x[1]["pipeline_tsr"]):
            print(f"  {nm:<30} {mt['pipeline_tsr']:>7.2%} {mt['rsr']:>7.2%} {mt['aat']:>7.2f}")

    with open(os.path.join(args.output_dir, f"{args.domain}_summary.json"), "w") as f:
        json.dump({n:{k:v for k,v in m.items()
                      if isinstance(v,(str,int,float,bool,type(None),dict,list))}
                   for n,m in all_m.items()}, f, indent=2, default=str)
    print(f"\nDone.  Total LLM calls: {llm.total_calls}")


if __name__ == "__main__":
    main()
