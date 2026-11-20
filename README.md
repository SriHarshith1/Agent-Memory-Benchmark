# Agent Memory Benchmark

An open benchmark for long-term memory in AI agents. It compares **Mem0** and **Zep (Graphiti)**, both open source and run locally, against two baselines: no memory and the full chat history.

Every system feeds the same answering agent, with the same prompt and model. Only the memory layer changes. Scoring is normalized exact match against a template answer key, with no LLM judge, so you can check every score by hand.

## What it tests

Each synthetic user has about 20 chat sessions. A few sessions contain facts and the rest are small talk. Then the agent answers 7 questions:

| Situation | Example | Right answer |
|---|---|---|
| Plain recall | "What did I name my rabbit?" | the pet's name |
| Someone else's fact | "Which city does my sister live in?" | the sister's city, not the user's |
| Updated fact | "Which city do I live in now?" | the new city, not the old one |
| Previous value | "Where did I live before?" | the old city |
| Deleted fact | "What was my old work email?" (after "please forget it") | `UNKNOWN` |
| Never mentioned | "What's my blood type?" | `UNKNOWN` |

## Results (preliminary: 3 users × 7 questions, 1 run)

| System | Accuracy | Precision | Recall | Deleted fact | Ingest / session |
|---|---|---|---|---|---|
| Full history (baseline) | 0.86 | 0.83 | 1.00 | 0/3 | — |
| Mem0 | 0.86 | 0.83 | 1.00 | 0/3 | 9.2 s |
| Zep (Graphiti) | 0.76 | 1.00 | 0.67 | 3/3 | 29 s |
| No memory (baseline) | 0.29 | 0.00 | 0.00 | 3/3 | — |

- **Forgetting:** after the user asks to forget an email, Mem0 still returns it for all 3 users. Graphiti marks the old fact as invalid and never returns it.
- **Recall:** Mem0 recalls every fact. Graphiti stores the facts, but its search returns nothing for 4 of 15 answerable questions, often when the question's wording differs from the stored fact.
- **Precision:** Zep never gives a wrong value. When it fails, it says `UNKNOWN`.
- **Speed:** Mem0 ingests about 3× faster on this setup.

The sample is small. One question moves accuracy by about 5 points. Two runs on the same users differ by one or two answers per system (Mem0 updated fact 2/3 → 3/3, Zep deleted fact 2/3 → 3/3). A larger run with repeats will replace these numbers.

The full report, with every question, reply and retrieved memory, is live at **https://sriharshith1.github.io/Agent-Memory-Benchmark/** (source: [`site/index.html`](site/index.html)). The raw rows are in [`results/20261120-181403.jsonl`](results/20261120-181403.jsonl). The earlier run is in [`results/20261120-043741.jsonl`](results/20261120-043741.jsonl).

## Setup

All models run locally through [Ollama](https://ollama.com): `qwen3.5` for the agent and for fact extraction, `nomic-embed-text` for embeddings.

```bash
ollama pull qwen3.5 && ollama pull nomic-embed-text
python -m venv venv && ./venv/bin/pip install -r requirements.txt

python generate.py                      # 10 users x 20 sessions -> data/generated/
python eval.py --systems none full mem0 zep --data "data/generated/*.json"
python report.py                        # builds site/index.html from the newest results file
```

Use `--runs 3` on `eval.py` to see run-to-run spread.

## Files

| File | Job |
|---|---|
| `generate.py` | Builds synthetic users from templates, with an exact answer key and self-checks |
| `memory.py` | The adapter interface (`reset`, `ingest`, `retrieve`) and the two baselines |
| `mem0_memory.py` | Mem0 OSS adapter (Ollama + local Qdrant) |
| `graphiti_memory.py` | Zep's Graphiti adapter (Ollama + embedded Kuzu) |
| `agent.py` | The fixed answering agent |
| `eval.py` | Runs every system, scores the replies, saves every row |
| `report.py` | Renders the static report page |

## Limits

- These are the open-source versions on a small local model. Mem0 Platform and Zep Cloud use their own models and can score differently.
- Graphiti runs on Kuzu, which Graphiti marks as deprecated. The adapter patches two driver bugs.
- The agent prompt has no rule about forget requests, so the deleted-fact row measures the memory system alone.
