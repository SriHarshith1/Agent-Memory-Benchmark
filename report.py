"""Build the results page: results/*.jsonl -> site/index.html (one static file).

All tables and charts are written here as plain HTML, so the page shows its data even when
JavaScript is blocked. The small script in the template only adds the Charts/Table switch,
the answer filters, tooltips and the theme button.

Usage:
    python report.py                                  # newest results file
    python report.py results/20261119-224435.jsonl    # a specific file (or several)
"""
import argparse
import glob
import json
import os
import statistics
import time
from collections import defaultdict
from html import escape

from eval import score

# Colour follows the system, never its rank. Slots 1-4 of the validated dataviz palette.
SYSTEMS = {
    "mem0": {"name": "Mem0", "kind": "vendor", "color": 1,
             "config": "mem0ai (open source, local) · Qdrant · extraction qwen3.5 · nomic-embed-text"},
    "zep": {"name": "Zep (Graphiti)", "kind": "vendor", "color": 2,
            "config": "graphiti-core 0.31, Zep's open-source engine · Kuzu · extraction qwen3.5 · nomic-embed-text"},
    "full": {"name": "Full history", "kind": "baseline", "color": 3,
             "config": "Every past message, in order, in the prompt. No retrieval."},
    "none": {"name": "No memory", "kind": "baseline", "color": 4,
             "config": "Stores nothing. Shows what the agent gets right with no memory."},
}
TYPES = {
    "plain_recall": ("Plain recall", "remember"),
    "other_person": ("Someone else's fact", "remember"),
    "updated_fact": ("Updated fact", "update"),
    "previous_value": ("Previous value", "update"),
    "deleted_fact": ("Deleted fact", "forget"),
    "never_mentioned": ("Never mentioned", "forget"),
}
METRICS = [
    # key, label, subtitle, lower_is_better, kind
    ("accuracy", "Accuracy", "correct ÷ all questions", False, "pct"),
    ("recall", "Recall", "facts it should know that it got", False, "pct"),
    ("precision", "Precision", "right when it gives an answer", False, "pct"),
    ("f1", "F1", "harmonic mean of precision and recall", False, "pct"),
    ("abstain", "Abstain", "says UNKNOWN when it should", False, "pct"),
    ("stale", "Stale answers", "replies with the outdated value", True, "pct"),
    ("ingest_s", "Ingest time", "median seconds per session", True, "sec"),
    ("retrieve_ms", "Retrieve time", "median ms per search", True, "ms"),
]


def fmt(v, kind):
    if v is None:
        return "—"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "sec":
        return "<0.01 s" if v < 0.01 else f"{v:.2f} s"
    return "<1 ms" if v < 1 else f"{v:.0f} ms"


def aggregate(rows):
    by_system = defaultdict(list)
    for r in rows:
        by_system[r["system"]].append(r)
    systems = []
    for sid, srows in by_system.items():
        info = SYSTEMS.get(sid, {"name": sid, "kind": "vendor", "color": 5, "config": ""})
        runs = defaultdict(list)
        for r in srows:
            runs[r["run"]].append(r)
        per_run = [score(rr) for rr in runs.values()]
        values, sds = {}, {}
        for key, *_ in METRICS[:6]:
            vals = [s[key] for s in per_run]
            values[key] = statistics.mean(vals)
            sds[key] = statistics.stdev(vals) if len(vals) > 1 else None
        ingest = [r["ingest_s_per_session"] for r in srows if "ingest_s_per_session" in r]
        retrieve = [r["retrieve_ms"] for r in srows if "retrieve_ms" in r]
        values["ingest_s"] = statistics.median(ingest) if ingest else None
        values["retrieve_ms"] = statistics.median(retrieve) if retrieve else None
        by_type = {}
        for t in TYPES:
            trows = [r for r in srows if r["type"] == t]
            if trows:
                by_type[t] = (sum(r["correct"] for r in trows), len(trows))
        systems.append({"id": sid, **info, "values": values, "sds": sds, "by_type": by_type,
                        "runs": len(runs), "n": len(srows) // max(len(runs), 1)})
    # Table order: vendors alphabetical, then baselines.
    systems.sort(key=lambda s: (s["kind"] != "vendor", s["name"].lower()))
    return systems


def best(systems, key, lower):
    pool = [s for s in systems if s["kind"] == "vendor" and s["values"].get(key) is not None]
    if not pool:
        return None
    return (min if lower else max)(pool, key=lambda s: s["values"][key])


def swatch(s):
    return f'<i class="sw c{s["color"]}" aria-hidden="true"></i>'


def deleted_acc(s):
    k, n = s["by_type"].get("deleted_fact", (0, 0))
    return k / n if n else None


# ---------- page blocks ----------

def hero(systems, meta):
    vendors = [s for s in systems if s["kind"] == "vendor"]
    none = next((s for s in systems if s["id"] == "none"), None)
    full = next((s for s in systems if s["id"] == "full"), None)
    # Group each system's wins into one sentence: "Mem0 leads accuracy at 81.0% and recall at 93.3%."
    wins = defaultdict(list)
    for key, label, _, lower, kind in METRICS[:6]:
        w = best(systems, key, lower)
        if w and key in ("accuracy", "recall", "precision", "stale"):
            text = (f"has the fewest stale answers at {fmt(w['values'][key], kind)}" if key == "stale"
                    else f"{label.lower()} at {fmt(w['values'][key], kind)}")
            wins[w["name"]].append(text)
    lead = []
    for name, parts in wins.items():
        leads = [p for p in parts if not p.startswith("has ")]
        other = [p for p in parts if p.startswith("has ")]
        clauses = ([f"leads {' and '.join(leads)}"] if leads else []) + other
        lead.append(f"{escape(name)} {' and '.join(clauses)}.")
    leaked = [s["name"] for s in vendors if (deleted_acc(s) or 0) < 1]
    if leaked:
        lead.append(f"{' and '.join(map(escape, leaked))} still return a deleted fact for some users.")
    lead.append(f"Independent benchmark of long-term memory for AI agents, on {meta['n_users']} synthetic users and "
                f"{meta['n_questions']} questions per system. Open source. Last measured {meta['date']}.")

    none_txt = (f"With no memory, the agent scores {fmt(none['values']['accuracy'], 'pct')} accuracy, and every point "
                f"comes from correctly saying UNKNOWN. Its recall is {fmt(none['values']['recall'], 'pct')}." if none else "")
    full_txt = (f"Putting the whole history in the prompt scores {fmt(full['values']['accuracy'], 'pct')}, but the prompt "
                f"grows with every session, and a long history can bury a forget request." if full else "")

    cards = "".join(
        f'<a class="card" href="#situations"><span class="idx">[0{i}]</span><b>{t}</b><span>{d}</span></a>'
        for i, (t, d) in enumerate([
            ("Remember", "Recall a fact about the user, and keep other people's facts apart."),
            ("Keep up", "Give the current value after a change, and the old one when asked."),
            ("Forget", "Drop a fact the user asked to delete. Admit what was never said."),
        ], 1))
    stats = "".join(f'<div class="stat"><span class="eyebrow">{k}</span><b>{v}</b></div>' for k, v in [
        ("No-memory accuracy", fmt(none["values"]["accuracy"], "pct") if none else "—"),
        ("Situations", len(TYPES)), ("Systems", len(systems)), ("Users", meta["n_users"]),
    ])
    grid = "".join(
        f'<div class="vendor">{swatch(s)}<span>{escape(s["name"])}</span>'
        f'<small>{"memory layer" if s["kind"] == "vendor" else "baseline"}</small></div>' for s in systems)

    return f"""
<div class="crumb">benchmarks / <b>agent-memory</b></div>
<div class="hero">
  <div class="hero-main">
    <div class="chip"><i class="sq"></i>{len(TYPES)} situations · {len(systems)} systems · last measured {meta['date']}</div>
    <h1>Agent Memory Benchmark</h1>
    <p>{' '.join(lead)}</p>
    <p><b>Why we built this.</b> A memory layer has three jobs: remember what the user said, keep up when it changes,
    and forget what the user asks it to forget. Public comparisons of memory vendors argue about judging method.
    This board uses template data with an exact answer key, so every score can be checked by hand.</p>
    <p><b>Model-only baseline.</b> {none_txt} {full_txt}</p>
    <p><b>How it is measured.</b> The answering agent, its prompt and the scorer stay the same for every system,
    so the comparison isolates the memory layer.</p>
  </div>
  <aside class="hero-side">
    <p><b>How to read this.</b> Start from the situation that matches your product, then read that row.</p>
    <ul>
      <li><b>Remember.</b> Read <a href="#situations">plain recall</a> and <a href="#situations">someone else's fact</a>.</li>
      <li><b>Keep up.</b> Read <a href="#situations">updated fact</a>. A stale answer counts as a failure.</li>
      <li><b>Forget.</b> Read <a href="#situations">deleted fact</a>. The right answer is UNKNOWN.</li>
    </ul>
    <p class="small">{meta['date']} · first run with Mem0 and Zep (Graphiti) on {meta['n_users']} users.</p>
  </aside>
</div>
<div class="cards">{cards}</div>
<div class="stats">{stats}</div>
<div class="vendors">{grid}</div>"""


def winners(systems):
    lines = []
    for key, label, _, lower, kind in [METRICS[0], METRICS[5], METRICS[6]]:
        w = best(systems, key, lower)
        if w:
            phrase = {"accuracy": "is most accurate at", "stale": "has the fewest stale answers at",
                      "ingest_s": "is fastest to ingest at"}[key]
            lines.append(f'<li>{swatch(w)}{escape(w["name"])} {phrase} {fmt(w["values"][key], kind)}</li>')
    forget = [s for s in systems if s["kind"] == "vendor" and deleted_acc(s) is not None]
    if forget:
        w = max(forget, key=deleted_acc)
        lines.append(f'<li>{swatch(w)}{escape(w["name"])} forgets best at {fmt(deleted_acc(w), "pct")}</li>')
    return f'<ul class="winners">{"".join(lines)}</ul>'


def charts(systems, n_q):
    out = []
    for key, label, sub, lower, kind in METRICS:
        have = [s for s in systems if s["values"].get(key) is not None]
        if not have:
            continue
        ranked = sorted(have, key=lambda s: s["values"][key], reverse=not lower)
        top = 1.0 if kind == "pct" else max(s["values"][key] for s in have) or 1
        bars = []
        for s in ranked:
            v = s["values"][key]
            h = max(0.5, v / top * 100)
            tip = f'{s["name"]} · {label}: {fmt(v, kind)}' + (" (baseline)" if s["kind"] == "baseline" else "")
            bars.append(f'''<div class="col" data-tip="{escape(tip)}">
  <div class="bar-area"><span class="val">{fmt(v, kind)}</span><div class="bar c{s["color"]}" style="height:{h:.1f}%"></div></div>
  <div class="who">{swatch(s)}<b>{escape(s["name"])}</b><small>{"baseline" if s["kind"] == "baseline" else "oss · local"}</small></div>
</div>''')
        arrow = "↓" if lower else "↑"
        nline = f"n = {n_q}" if kind == "pct" else "wall time"
        out.append(f'''<figure class="chart">
  <figcaption><i class="sq"></i><b>{label}</b><span>{nline} · {sub} · {arrow}</span></figcaption>
  <div class="plot">{"".join(bars)}</div>
</figure>''')
    return f'<div class="charts" id="view-charts">{"".join(out)}</div>'


def table(systems):
    head = "".join(f"<th>{label}</th>" for _, label, *_ in METRICS)
    body = []
    for s in systems:
        cells = []
        for key, label, _, lower, kind in METRICS:
            v, sd = s["values"].get(key), s["sds"].get(key)
            w = best(systems, key, lower)
            cls = " best" if w and w["id"] == s["id"] else ""
            sd_txt = f' <span class="sd">± {sd * 100:.1f}</span>' if sd is not None else ""
            cells.append(f'<td class="num{cls}">{fmt(v, kind)}{sd_txt}</td>')
        tag = '<span class="tag">baseline</span>' if s["kind"] == "baseline" else ""
        body.append(f'<tr><td class="sys">{swatch(s)}{escape(s["name"])}{tag}</td>'
                    f'<td class="cfg">{escape(s["config"])}</td>{"".join(cells)}</tr>')
    return f'''<div class="table-wrap" id="view-table" hidden><table>
<caption>Memory systems on identical users and questions. Vendors alphabetical, then baselines. ▲ best vendor value per column.</caption>
<thead><tr><th>System</th><th class="l">Configuration</th>{head}</tr></thead>
<tbody>{"".join(body)}</tbody></table></div>'''


def situations(systems):
    head = "".join(f"<th>{escape(TYPES[t][0])}</th>" for t in TYPES)
    body = []
    for s in systems:
        cells = []
        for t in TYPES:
            k, n = s["by_type"].get(t, (0, 0))
            if not n:
                cells.append('<td class="cell">—</td>')
                continue
            acc = k / n
            tip = f'{s["name"]} · {TYPES[t][0]}: {k}/{n} correct'
            cells.append(f'<td class="cell" style="--a:{acc * 38:.0f}%" data-tip="{escape(tip)}">'
                         f'{acc * 100:.0f}%<small>{k}/{n}</small></td>')
        body.append(f'<tr><td class="sys">{swatch(s)}{escape(s["name"])}</td>{"".join(cells)}</tr>')
    return f'<div class="table-wrap"><table class="heat"><thead><tr><th>System</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def explorer(rows, systems):
    names = {s["id"]: s for s in systems}
    opts_sys = "".join(f'<option value="{s["id"]}">{escape(s["name"])}</option>' for s in systems)
    opts_type = "".join(f'<option value="{t}">{escape(v[0])}</option>' for t, v in TYPES.items())
    items = []
    for r in rows:
        s = names.get(r["system"], {"name": r["system"], "color": 5})
        stale = r.get("stale_answer")
        stale_hit = bool(stale) and r["reply"].strip().lower().rstrip(".!") == stale.lower()
        mems = r.get("memories")
        if mems is None:
            mem_html = '<div class="mem-note">memories not saved in this run</div>'
        elif r["system"] == "full":
            mem_html = f'<div class="mem-note">{len(mems)} messages passed (full history)</div>'
        else:
            lis = "".join(f"<li>{escape(m)}</li>" for m in mems[:15]) or "<li>(nothing returned)</li>"
            mem_html = f'<details><summary>{len(mems)} memories returned</summary><ul>{lis}</ul></details>'
        items.append(f'''<div class="qrow" data-sys="{r["system"]}" data-kind="{s.get("kind", "")}" data-type="{r["type"]}" data-ok="{int(r["correct"])}">
  <div class="qhead"><span class="verdict {"ok" if r["correct"] else "bad"}">{"✓ pass" if r["correct"] else "✗ fail"}</span>
    <span>{swatch(s)}{escape(s["name"])}</span><span>{escape(TYPES.get(r["type"], (r["type"],))[0])}</span>
    <span>user {escape(r["user_id"])} · run {r["run"]}</span>{'<span class="verdict bad">stale value</span>' if stale_hit else ""}</div>
  <div class="qtext">{escape(r["question"])}</div>
  <div class="qans"><span>expected <b>{escape(r["expected"])}</b></span><span>got <b>{escape(r["reply"])}</b></span>{f"<span>outdated value {escape(stale)}</span>" if stale else ""}</div>
  {mem_html}
</div>''')
    return f'''<div class="toolbar">
  <select id="fSys" aria-label="System"><option value="">All systems</option><option value="vendors">Mem0 + Zep</option>{opts_sys}</select>
  <select id="fType" aria-label="Situation"><option value="">All situations</option>{opts_type}</select>
  <select id="fOut" aria-label="Outcome"><option value="all">All answers</option><option value="0">Failures only</option><option value="1">Passes only</option></select>
  <span class="count" id="fCount">{len(rows)} rows</span>
</div>
<div class="qlist" id="qlist">{"".join(items)}</div>'''


def build_body(rows, files):
    systems = aggregate(rows)
    users = {r["user_id"] for r in rows}
    n_q = max((s["n"] for s in systems), default=0)
    runs = max((s["runs"] for s in systems), default=1)
    meta = {"n_users": len(users), "n_questions": n_q, "runs": runs,
            "date": time.strftime("%d %b %Y", time.localtime(max(os.path.getmtime(f) for f in files)))}
    return f"""
{hero(systems, meta)}

<section id="overall">
  <div class="eyebrow-line"><span class="idx">[01]</span> OVERALL</div>
  <h2>Overall board</h2>
  <p class="desc">Every system answers the same questions about the same users. Systems are listed alphabetically in the table;
  charts sort each metric best-first. No single metric decides a ranking.</p>
  <div class="issue">
    <div class="issue-tag"><i class="sq"></i>PRELIMINARY · SMALL SAMPLE · FULL RUN PENDING</div>
    <p><b>These results come from {meta['n_users']} users × {n_q // max(meta['n_users'], 1)} questions, {runs} run{'s' if runs > 1 else ''}.</b>
    One question moves accuracy by about {100 / max(n_q, 1):.0f} points. A larger run with repeats will replace these numbers.</p>
    <p class="note">Until then, treat differences under ~10 points as noise.</p>
  </div>
  {winners(systems)}
  <div class="bar-head"><span class="eyebrow">ALL SYSTEMS PER METRIC</span>
    <div class="seg" role="group" aria-label="View"><button type="button" data-view="charts" aria-pressed="true">CHARTS</button><button type="button" data-view="table" aria-pressed="false">TABLE</button></div></div>
  {charts(systems, n_q)}
  {table(systems)}
  <div class="boxed"><span class="tag2">OPEN CODE</span><span>The data generator, adapters, agent and scorer are on GitHub: <a href="https://github.com/SriHarshith1/Agent-Memory-Benchmark">SriHarshith1/Agent-Memory-Benchmark</a>. Every row on this page is in the repo, so you can check each score.</span></div>
  <details class="acc"><summary><span>[01.a]</span> methodology and metric definitions</summary><div>
    <h3>How the board is built</h3>
    <ol>
      <li>Generate users from templates with a fixed seed. Every answer key is exact.</li>
      <li>For each system: <code>reset</code> the user, then <code>ingest</code> each session in time order.</li>
      <li>For each question: <code>retrieve</code> memories, then pass them to one fixed agent ({escape('qwen3.5 via Ollama, temperature 0, thinking off')}).</li>
      <li>The agent replies with a short answer or exactly <code>UNKNOWN</code>. Scoring is normalized exact match. No LLM judge.</li>
      <li>The agent is <b>not</b> told to respect forget requests. Deleting a fact is the memory system's job, so the deleted-fact row measures the memory system alone.</li>
    </ol>
    <h3>What each metric means</h3>
    <ul>
      <li><b>Accuracy</b> · correct ÷ all questions. UNKNOWN on an unanswerable question counts as correct.</li>
      <li><b>Precision</b> · correct answers ÷ all non-UNKNOWN answers.</li>
      <li><b>Recall</b> · correct answers ÷ questions that have a real answer.</li>
      <li><b>F1</b> · harmonic mean of precision and recall.</li>
      <li><b>Abstain</b> · UNKNOWN replies ÷ questions whose answer is UNKNOWN.</li>
      <li><b>Stale answers</b> · replies equal to the outdated value ÷ questions that have one. Lower is better.</li>
      <li><b>Ingest / retrieve time</b> · wall time on one Mac with local models. Compare systems only within this board.</li>
    </ul>
  </div></details>
</section>

<section id="situations">
  <div class="eyebrow-line"><span class="idx">[02]</span> BY SITUATION</div>
  <h2>Accuracy by situation</h2>
  <p class="desc">Each question type tests one memory behaviour. The cell shade grows with accuracy. The fraction is correct ÷ total.</p>
  {situations(systems)}
  <details class="acc"><summary><span>[02.a]</span> how the dataset is built</summary><div>
    <p>Each user has a timeline of chat sessions. Four sessions carry facts, in this order. All other sessions are small talk.</p>
    <ol>
      <li><b>Intro</b> · pet name, home city, work email.</li>
      <li><b>Sibling</b> · the sibling's name and city (another person's facts).</li>
      <li><b>Move</b> · the user moves to a new city and a new company.</li>
      <li><b>Forget</b> · the user asks the assistant to forget the old work email.</li>
    </ol>
    <p>Filler lines mention other cities as hard negatives. The old city, new city and sibling city are always different.
    A self-check rejects any user whose answer is missing from the text, or whose deleted email appears after the forget request.</p>
  </div></details>
</section>

<section id="answers">
  <div class="eyebrow-line"><span class="idx">[03]</span> EVERY ANSWER</div>
  <h2>Every answer</h2>
  <p class="desc">The rows behind the tables: the question, the answer key, the reply, and the memories the system returned.</p>
  {explorer(rows, systems)}
</section>

<footer><span>source: {escape(", ".join(os.path.basename(f) for f in files))}</span><span>CC-BY-4.0</span></footer>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    parser.add_argument("--out", default="site/index.html")
    args = parser.parse_args()

    files = args.files or sorted(glob.glob("results/*.jsonl"))[-1:]
    if not files:
        raise SystemExit("no results/*.jsonl found - run eval.py first")
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        raise SystemExit(f"results file not found: {', '.join(missing)}")
    rows = [json.loads(line) for f in files for line in open(f) if line.strip()]

    template = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "site_template.html")).read()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(template.replace("__BODY__", build_body(rows, files)))
    print(f"Wrote {args.out} from {', '.join(files)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
