"""Eval loop: run each memory system on each synthetic user, then score with exact match.

Usage:
    python eval.py                          # all systems, all users in data/
    python eval.py --systems none full      # only the baselines (no Mem0 needed)
    python eval.py --runs 3                 # repeat to see run-to-run variance

Metrics (no LLM judge, only normalized exact match):
    "answerable" question = its key is a real value.  "unanswerable" = its key is UNKNOWN.
    "attempted"           = the agent gave a value, not UNKNOWN.

    accuracy  = correct / all questions             (UNKNOWN on an unanswerable question counts as correct)
    precision = correct attempts / all attempts     -> when it gives an answer, how often is it right?
    recall    = correct attempts / answerable       -> of the facts it should know, how many did it get?
    f1        = 2 * P * R / (P + R)
    abstain   = UNKNOWN replies / unanswerable      -> does it say "I don't know" when it should?
    stale     = replies equal to stale_answer / questions that have one  -> lower is better
"""
import argparse
import glob
import json
import os
import re
import time
from collections import defaultdict

import ollama

from agent import UNKNOWN, answer
from memory import make_system


def normalize(text: str) -> str:
    """Lowercase, remove punctuation at the ends and extra spaces. 'Bangalore.' -> 'bangalore'."""
    text = text.strip().lower()
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    return re.sub(r"\s+", " ", text)


def is_match(reply: str, expected: str) -> bool:
    return normalize(reply) == normalize(expected)


def score(rows: list[dict]) -> dict:
    n = len(rows)
    answerable = [r for r in rows if r["expected"] != UNKNOWN]
    unanswerable = [r for r in rows if r["expected"] == UNKNOWN]
    attempted = [r for r in rows if not is_match(r["reply"], UNKNOWN)]
    correct_attempts = [r for r in attempted if r["correct"]]
    with_stale = [r for r in rows if r.get("stale_answer")]

    p = len(correct_attempts) / len(attempted) if attempted else 0.0
    rc = len(correct_attempts) / len(answerable) if answerable else 0.0
    return {
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n if n else 0.0,
        "precision": p,
        "recall": rc,
        "f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
        "abstain": sum(r["correct"] for r in unanswerable) / len(unanswerable) if unanswerable else 0.0,
        "stale": sum(is_match(r["reply"], r["stale_answer"]) for r in with_stale) / len(with_stale) if with_stale else 0.0,
    }


def run_system(name: str, users: list[dict], run: int) -> list[dict]:
    system = make_system(name)
    rows = []
    for user in users:
        user_id = f"{user['user_id']}-run{run}"  # new id per run, so no data leaks between runs
        system.reset(user_id)
        t0 = time.perf_counter()
        for session in user["sessions"]:
            system.ingest(user_id, session)
        ingest_s = (time.perf_counter() - t0) / len(user["sessions"])

        for q in user["questions"]:
            t0 = time.perf_counter()
            memories = system.retrieve(user_id, q["question"])
            retrieve_ms = (time.perf_counter() - t0) * 1000
            reply = answer(q["question"], memories)
            rows.append({
                "system": name,
                "run": run,
                "user_id": user["user_id"],
                "type": q["type"],
                "question": q["question"],
                "expected": q["answer"],
                "stale_answer": q.get("stale_answer"),
                "reply": reply,
                "correct": is_match(reply, q["answer"]),
                "n_memories": len(memories),
                "memories": memories,
                "ingest_s_per_session": round(ingest_s, 3),
                "retrieve_ms": round(retrieve_ms, 1),
            })
            mark = "✓" if rows[-1]["correct"] else "✗"
            print(f"  {mark} [{name}] {q['type']:<16} expected={q['answer']!r:<14} got={reply!r}")
    return rows


def print_table(title: str, groups: dict[str, list[dict]]) -> None:
    
    cols = ["n", "accuracy", "precision", "recall", "f1", "abstain", "stale"]
    print(f"\n{title}")
    print(f"{'':<18}" + "".join(f"{c:>10}" for c in cols))
    for key, rows in groups.items():
        s = score(rows)
        print(f"{key:<18}" + "".join(f"{s[c]:>10}" if c == "n" else f"{s[c]:>10.2f}" for c in cols))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", default=["none", "full", "mem0"])
    parser.add_argument("--data", default="data/*.json")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    try:
        ollama.list()
    except ConnectionError:
        raise SystemExit("Ollama is not running. Start it (open the Ollama app or run `ollama serve`), then run eval.py again.")

    paths = sorted(glob.glob(args.data))
    if not paths:
        raise SystemExit(f"no user files match {args.data!r}")
    users = [json.load(open(path)) for path in paths]
    print(f"{len(users)} users, {sum(len(u['questions']) for u in users)} questions, systems={args.systems}")

    all_rows = []
    for name in args.systems:
        for run in range(args.runs):
            print(f"\n== {name} (run {run}) ==")
            all_rows += run_system(name, users, run)

    # Save every row, so you can check each answer later.
    os.makedirs("results", exist_ok=True)
    out = f"results/{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    with open(out, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    by_system = defaultdict(list)
    by_type = defaultdict(list)
    for r in all_rows:
        by_system[r["system"]].append(r)
        by_type[f"{r['system']}/{r['type']}"].append(r)

    print_table("Overall", by_system)
    print("\nAccuracy by question type")
    for key, rows in by_type.items():
        print(f"  {key:<32} {score(rows)['accuracy']:.2f}  ({sum(r['correct'] for r in rows)}/{len(rows)})")
    print(f"\nSaved {len(all_rows)} rows to {out}")


if __name__ == "__main__":
    main()
