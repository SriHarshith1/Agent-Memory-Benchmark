"""Memory systems behind one interface.

Every memory system has the same 3 methods:
    reset(user_id)              -> delete everything stored for this user
    ingest(user_id, session)    -> store one session (a dict with "messages")
    retrieve(user_id, question) -> return the memories for this question, as list[str]

The harness (eval.py) only calls these 3 methods, so it does not care which system is behind them.
"""


class NoMemory:
    """Lower bound: remembers nothing. It can only pass the 'UNKNOWN' questions."""

    def reset(self, user_id: str) -> None:
        pass

    def ingest(self, user_id: str, session: dict) -> None:
        pass

    def retrieve(self, user_id: str, question: str) -> list[str]:
        return []


class FullHistory:
    """'Put everything in the prompt' baseline: returns every message, in order, for every question."""

    def __init__(self):
        self.history: dict[str, list[str]] = {}

    def reset(self, user_id: str) -> None:
        self.history[user_id] = []

    def ingest(self, user_id: str, session: dict) -> None:
        lines = self.history.setdefault(user_id, [])
        for msg in session["messages"]:
            lines.append(f"[{session['session_id']}] {msg['role']}: {msg['content']}")

    def retrieve(self, user_id: str, question: str) -> list[str]:
        return list(self.history.get(user_id, []))


def make_system(name: str):
    """Create a memory system by name. Vendor imports are lazy, so the baselines run without them."""
    if name == "none":
        return NoMemory()
    if name == "full":
        return FullHistory()
    if name == "mem0":
        from mem0_memory import Mem0Memory
        return Mem0Memory()
    if name == "zep":
        from graphiti_memory import GraphitiMemory  # Zep's open-source engine, run locally
        return GraphitiMemory()
    raise ValueError(f"unknown memory system: {name}")
